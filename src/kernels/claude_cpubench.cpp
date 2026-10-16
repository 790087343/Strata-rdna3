// CPU-only real-weight cache working-set and per-group cost sweep. No GPU calls.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <sys/prctl.h>
namespace c=strata::kernels::cpu;
uint64_t hash(const void*p,size_t n){uint64_t h=14695981039346656037ull;auto b=(const unsigned char*)p;while(n--){h^=*b++;h*=1099511628211ull;}return h;}
int main(int argc,char**argv){
 setvbuf(stdout,nullptr,_IONBF,0);
 
 if(argc<4){fprintf(stderr,"usage: replay shard.gguf layer working_set_MiB [workers]\n");return 2;}
 int layer=atoi(argv[2]),mib=atoi(argv[3]);if(layer<0||mib<32||mib>8192)return 2;
 std::vector<std::unique_ptr<strata::GgufFile>> files;
 const std::string first=argv[1];const auto marker=first.rfind("-00001-of-");
 const int shards=marker==std::string::npos?1:std::stoi(first.substr(marker+10,5));
 if(shards<1||shards>16)return 2;
 for(int i=1;i<=shards;++i){std::string path=first;if(marker!=std::string::npos){char n[6];std::snprintf(n,sizeof(n),"%05d",i);path.replace(marker+1,5,n);}files.emplace_back(new strata::GgufFile(path));}
 const strata::TensorInfo*t[3]={};const strata::GgufFile*owner[3]={};const char*roles[]={"gate","up","down"};
 for(const auto& file:files)for(const auto&ti:file->tensors())for(int r=0;r<3;++r)if(ti.name=="blk."+std::to_string(layer)+".ffn_"+roles[r]+"_exps.weight"){
  if(t[r]){fprintf(stderr,"Duplicate tensor role\n");return 2;}t[r]=&ti;owner[r]=file.get();
 }
 if(!t[0]||!t[1]||!t[2]){fprintf(stderr,"Layer roles missing in supplied GGUF; choose a shard/layer containing all three\n");return 2;}
 c::NativeFmt f;std::string err;if(!c::native_fmt((int)t[0]->type,(int)t[2]->type,c::H,c::FF,f,err)){fprintf(stderr,"%s\n",err.c_str());return 2;}
 if(t[0]->type!=t[1]->type)return 2;
 for(int r=0;r<3;++r)if(t[r]->shape.size()!=3 || t[r]->shape[0]!=(uint64_t)(r==2?c::FF:c::H) || t[r]->shape[1]!=(uint64_t)(r==2?c::H:c::FF) || t[r]->shape[2]==0){fprintf(stderr,"Unexpected tensor shape\n");return 2;}
 const int G=int((size_t(mib)*1024*1024+f.bytes-1)/f.bytes);
 const uint64_t experts=t[0]->shape[2];
 if((uint64_t)G>experts || experts>1000000)return 2;
 for(int r=0;r<3;++r){
  const uint64_t span=strata::tensor_payload_bytes(*t[r]);
  const uint64_t per=r==2?f.bytes-f.down_off:f.up_off;
  if(t[r]->shape[2]!=experts || !span || span/experts!=per || span%experts ||
     owner[r]->data_start()>owner[r]->file_size() || t[r]->offset>owner[r]->file_size()-owner[r]->data_start() ||
     span>owner[r]->file_size()-owner[r]->data_start()-t[r]->offset){fprintf(stderr,"Tensor byte-span mismatch\n");return 2;}
 }
 const size_t weight_bytes=size_t(G)*f.bytes, alignment=2u<<20;
 std::unique_ptr<uint8_t,decltype(&std::free)> storage((uint8_t*)std::aligned_alloc(alignment,((weight_bytes+alignment-1)/alignment)*alignment),&std::free);
 if(!storage){fprintf(stderr,"Weight allocation failed\n");return 2;}
 uint8_t* blobs=storage.get();
 for(int e=0;e<G;++e){uint64_t id=(uint64_t(e)*37+5)%experts;auto b=blobs+size_t(e)*f.bytes;memcpy(b,owner[0]->tensor_data(*t[0])+size_t(id)*f.up_off,f.up_off);memcpy(b+f.up_off,owner[1]->tensor_data(*t[1])+size_t(id)*f.up_off,f.up_off);memcpy(b+f.down_off,owner[2]->tensor_data(*t[2])+size_t(id)*(f.bytes-f.down_off),f.bytes-f.down_off);}
 std::mt19937 rng(781);std::vector<float>x(8*c::H);for(auto&v:x)v=float(int(rng()%65537)-32768)/16384.f;
 // 64-byte alignment for native SIMD activation storage.
 struct alignas(64) NativeAct{uint8_t bytes[c::kNativeActBytes];};
 std::vector<NativeAct>na(8);std::vector<c::ActQ>act(8);
 for(int t0=0;t0<8;++t0){if(c::q2_native_kernels(f.gu_type))c::act_quant_any(x.data()+t0*c::H,c::H,act[t0]);else c::native_quant_act(f,x.data()+t0*c::H,na[t0].bytes);}
 const auto topo=c::detect_cpu_topology(true);
 const int nw=argc>4?atoi(argv[4]):8;
 c::ExpertPool pool(nw,true,true);
 const auto previous=c::pin_current_thread(topo.host_core);
 if(!previous.valid){fprintf(stderr,"Host affinity failed\n");return 2;}
 std::vector<float>out(size_t(8)*8*c::H);std::vector<c::ExpertJobMulti>jobs(8);
 if(0)printf("{\"kind\":\"input\",\"layer\":%d,\"ring_experts\":%d,\"gu_type\":%d,\"down_type\":%d,\"workers\":%d,\"weight_bytes\":%zu,\"weights_hash\":\"%016llx\"}\n",layer,G,f.gu_type,f.d_type,pool.workers(),weight_bytes,(unsigned long long)hash(blobs,weight_bytes));
 // "parity": every expert of the ring (G of them), 6 activations (random, large, tiny, sparse), the new single-token
 // kernel against ggml-cpu's dot, gate/up SwiGLU output bit for bit (2026-10-15)
 if(argc>5 && std::string(argv[5])=="parity"){
  if(!((f.gu_type==21 && c::iq3s_1t_supported()) || (f.gu_type==18 && c::iq3xxs_1t_supported()))){printf("parity: skip type %d\n",f.gu_type);return 0;}
  const int FF=(int)f.n_ff;std::vector<float> a(FF),b(FF);float* fa=a.data();
  struct alignas(64) NA{uint8_t b[c::kNativeActBytes];};std::vector<NA> acts(6);
  std::mt19937 r2(12345);std::vector<float> xv(c::H);
  for(int k=0;k<6;++k){for(int i=0;i<(int)c::H;++i){float v=float(int(r2()%65537)-32768)/16384.f;
    if(k==1)v*=1e4f; if(k==2)v*=1e-4f; if(k==3 && (r2()%8))v=0; if(k==4)v=(i%97==0)?127.f:-0.01f*(i%5); if(k==5)v=std::sin(i*0.37f)*3;xv[i]=v;}
   c::native_quant_act(f,xv.data(),acts[k].b);}
  long long rows=0,same=0;
  for(int e=0;e<G;++e)for(int k=0;k<6;++k){
   const uint8_t* blob=blobs+size_t(e)*f.bytes;
   // the reference: ggml-cpu's own dot for every gate and up row (native_gu_rows would take the new kernel)
   {const auto dot=ggml_get_type_traits_cpu((ggml_type)f.gu_type)->vec_dot;
    for(int r=0;r<FF;++r){float g=0,u=0;dot((int)f.n_embd,&g,0,blob+size_t(r)*f.gu_row,0,acts[k].b,0,1);
     dot((int)f.n_embd,&u,0,blob+f.up_off+size_t(r)*f.gu_row,0,acts[k].b,0,1);fa[r]=(g/(1.f+std::exp(-g)))*u;}}
   if(f.gu_type==21)c::iq3s_gu_rows_1t(blob,f.gu_row,f.up_off,(int)f.n_embd,acts[k].b,b.data(),0,FF);
   else c::iq3xxs_gu_rows_1t(blob,f.gu_row,f.up_off,(int)f.n_embd,acts[k].b,b.data(),0,FF);
   for(int i=0;i<FF;++i){++rows;same+=std::memcmp(&a[i],&b[i],4)==0;}
   for(int i=0;i<FF && f.gu_type==21;i+=97){float o;c::iq3s_rows_1t(blob,f.gu_row,(int)f.n_embd,acts[k].b,&o-i,i,i+1);
    float g=0;ggml_get_type_traits_cpu((ggml_type)21)->vec_dot((int)f.n_embd,&g,0,blob+size_t(i)*f.gu_row,0,acts[k].b,0,1);++rows;same+=std::memcmp(&o,&g,4)==0;}
  }
  printf("parity layer %d: %d experts x 6 activations: %lld/%lld rows bitwise\n",layer,G,same,rows);
  return same==rows?0:4;
 }
 // "dot": one core, one token, one warm expert: the gate/up rows (and the down rows) through the production dispatch
 // (ggml-cpu's dot at one token) against the AVX-512 kernel at one token, outputs compared bit for bit (2026-10-15)
 if(argc>5 && std::string(argv[5])=="dot"){
  const uint8_t* blob=blobs;const int FF=(int)f.n_ff;const int reps=argc>6?atoi(argv[6]):200;
  std::vector<float> a(FF),b(FF);float* fa=a.data();float* fb=b.data();
  const void* ap[1]={na[0].bytes};
  auto time=[&](auto&& fn){fn();auto t0=std::chrono::steady_clock::now();for(int r=0;r<reps;++r)fn();return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/reps;};
  const double tg=time([&]{c::native_gu_rows(f,blob,ap,1,&fa,0,FF);});
  double tx=-1;
  if(c::iq512_supported(f.gu_type)) tx=time([&]{c::iq512_gu_rows(f.gu_type,blob,f.gu_row,f.up_off,(int)f.n_embd,ap,1,&fb,0,FF);});
  if(f.gu_type==21 && c::iq3s_1t_supported() && getenv("BENCH_IQ3S_1T")) tx=time([&]{c::iq3s_gu_rows_1t(blob,f.gu_row,f.up_off,(int)f.n_embd,na[0].bytes,fb,0,FF);});
  if(f.gu_type==18 && c::iq3xxs_1t_supported() && getenv("BENCH_IQ3S_1T")) tx=time([&]{c::iq3xxs_gu_rows_1t(blob,f.gu_row,f.up_off,(int)f.n_embd,na[0].bytes,fb,0,FF);});
  int same=0;double maxrel=0;for(int i=0;i<FF;++i){same+=std::memcmp(&a[i],&b[i],4)==0;maxrel=std::max(maxrel,(double)std::fabs(a[i]-b[i])/(std::fabs(a[i])+1e-6));}
  const double gub=2.0*f.up_off;
  printf("dot gu type %d: ggml %.1f us (%.2f GB/s/core)  iq512 nt1 %.1f us (%.2f GB/s/core)  bitwise %d/%d max rel %.2g\n",f.gu_type,tg,gub/tg/1e3,tx,tx>0?gub/tx/1e3:0.0,tx>0?same:0,FF,tx>0?maxrel:0.0);
  // down rows: one token's h (n_ff) quantized as the pool does
  std::vector<float> h(FF);for(int i=0;i<FF;++i)h[i]=a[i];
  struct alignas(64) H{uint8_t b[c::kNativeHBytes];};H hq;c::native_quant_h(f,h.data(),hq.b);
  const void* hp[1]={hq.b};std::vector<float> o(c::H);float* op=o.data();
  const double td=time([&]{c::native_down_rows(f,blob,hp,1,&op,0,(int)c::H);});
  printf("dot down type %d: ggml %.1f us (%.2f GB/s/core)\n",f.d_type,td,(double)(f.bytes-f.down_off)/td/1e3);
  return 0;
 }
 // "warm": a call's experts cold (rotating through the layer, far beyond L3) against just read by the pool
 // (all of them, or the first half) - what a prefetch during the GPU phase would buy (2026-10-15)
 if(argc>5 && std::string(argv[5])=="warm"){
  for(int groups:{2,4,6,8})for(int nt:{1,2}){
   for(int e=0;e<groups;++e){auto&j=jobs[e];j.nt=nt;for(int t0=0;t0<nt;++t0){j.act[t0]=&act[t0];j.nact[t0]=na[t0].bytes;j.out[t0]=out.data()+(size_t(e)*8+t0)*c::H;}}
   const int calls=(G+groups-1)/groups;
   for(int round=0;round<3;++round){
    double cold=0,warm=0,half=0;int64_t ncalls=0;
    for(int i=0;i<calls;++i,++ncalls){
     for(int e=0;e<groups;++e)jobs[e].blob=blobs+size_t((i*groups+e)%G)*f.bytes;
     auto t0=std::chrono::steady_clock::now();pool.run_split_multi_native(f,jobs.data(),groups);
     cold+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
    }
    for(int i=0;i<calls;++i){
     for(int e=0;e<groups;++e)jobs[e].blob=blobs+size_t((i*groups+e+G/2)%G)*f.bytes;
     pool.run_split_multi_native(f,jobs.data(),groups);
     auto t0=std::chrono::steady_clock::now();pool.run_split_multi_native(f,jobs.data(),groups);
     warm+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
    }
    for(int i=0;i<calls;++i){
     for(int e=0;e<groups;++e)jobs[e].blob=blobs+size_t((i*groups+e+G/4)%G)*f.bytes;
     pool.run_split_multi_native(f,jobs.data(),groups/2);
     auto t0=std::chrono::steady_clock::now();pool.run_split_multi_native(f,jobs.data(),groups);
     half+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count();
    }
    for(float v:out)if(!std::isfinite(v))return 3;
    printf("warm groups %d nt %d round %d: cold %.1f us  warm %.1f us (%.2fx)  half-warm %.1f us (%.2fx)  cold %.1f GB/s\n",groups,nt,round,
           cold/ncalls,warm/ncalls,cold/warm,half/ncalls,cold/half,(double)f.bytes*groups/(cold/ncalls)/1e3);
   }
  }
  c::restore_thread_affinity(previous);
  puts("{\"kind\":\"complete\",\"status\":\"pass\"}");
  return 0;
 }
 for(int groups:{1,3,6})for(int nt:{1,2}){
  for(int e=0;e<groups;++e){auto&j=jobs[e];j.nt=nt;for(int t0=0;t0<nt;++t0){j.act[t0]=&act[t0];j.nact[t0]=na[t0].bytes;j.out[t0]=out.data()+(size_t(e)*8+t0)*c::H;}}
  const int calls=(G+groups-1)/groups;
  for(int round=-1;round<2;++round){
   auto begin=std::chrono::steady_clock::now();
   for(int rep=0;rep<4;++rep)for(int i=0;i<calls;++i){
    for(int e=0;e<groups;++e)jobs[e].blob=blobs+size_t((i*groups+e)%G)*f.bytes;
    pool.run_split_multi_native(f,jobs.data(),groups);
   }
   const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
   for(float v:out)if(!std::isfinite(v))return 3;
   if(round>=0)printf("groups %d nt %d: %.1f us/batch %.1f us/expert %.1f GB/s\n",groups,nt,us/(4*calls),us/(4*calls*groups),(double)f.bytes*groups/(us/(4*calls))/1e3);
  }
 }
 c::restore_thread_affinity(previous);
 puts("{\"kind\":\"complete\",\"status\":\"pass\"}");
}

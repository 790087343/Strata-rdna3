// CPU-only real-weight cache working-set and per-group cost sweep. No GPU calls.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

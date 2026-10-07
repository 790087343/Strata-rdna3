// aibox RDNA MMVQ numerical/column consistency screen and event-timed microbenchmark.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/hip_compat/cuda_runtime.h"
#include "ggml.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <fstream>
using namespace strata::kernels;
void ck(cudaError_t x) { if(x!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
struct Dev {void* p; Dev(size_t n){ck(cudaMalloc(&p,n));} ~Dev(){cudaFree(p);} };
int main(int argc, char** argv){try{
 std::ofstream dump; if(argc==2){dump.open(argv[1],std::ios::binary);if(!dump)throw std::runtime_error("output file");}
 cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
 struct Shape{ggml_type type;int n,m;};
 for(auto sh:std::vector<Shape>{{GGML_TYPE_Q8_0,2560,2560},{GGML_TYPE_Q8_0,6144,2560},{GGML_TYPE_Q8_0,2560,10240},{GGML_TYPE_Q8_0,2560,12288},{GGML_TYPE_Q8_0,2560,512},{GGML_TYPE_Q8_0,2560,513},{GGML_TYPE_Q8_0,2560,1025},{GGML_TYPE_Q4_0,2560,512},{GGML_TYPE_Q5_0,2560,512},{GGML_TYPE_Q3_K,2560,512},{GGML_TYPE_IQ4_XS,2560,512},{GGML_TYPE_Q8_0,2560,640},{GGML_TYPE_Q8_0,640,2560},{GGML_TYPE_Q6_K,2560,4096},{GGML_TYPE_Q4_K,2560,1280},{GGML_TYPE_Q5_K,2560,1280},{GGML_TYPE_Q4_0,640,1025},{GGML_TYPE_Q5_0,640,1025},{GGML_TYPE_IQ4_NL,640,1025},{GGML_TYPE_Q2_0,2560,513}}){
  int n=sh.n,m=sh.m;auto tr=ggml_get_type_traits(sh.type);size_t rb=ggml_row_size(sh.type,n);
  std::vector<unsigned char>w(rb*m);std::vector<float>f(n),deq((size_t)n*m),x((size_t)n*8);
  for(int r=0;r<m;r++){for(int k=0;k<n;k++)f[k]=.03f*std::sin(k*.173f+r*.117f)+.001f*((k*37+r*13)%51-25);tr->from_float_ref(f.data(),w.data()+r*rb,n);tr->to_float(w.data()+r*rb,deq.data()+(size_t)r*n,n);}
  for(size_t k=0;k<x.size();k++)x[k]=.7f*std::sin(k*.091f)+.2f*std::cos(k*.037f);
  Dev dw(w.size()),dx(x.size()*4),dq(native_q8_1_bytes(n,8)),dy((size_t)m*8*4),ds((size_t)m*8*4);
  ck(cudaMemcpy(dw.p,w.data(),w.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
  native_quantize_q8_1((float*)dx.p,dq.p,n,8,s);ck(cudaStreamSynchronize(s));
  std::vector<float>ref((size_t)m*8),got(ref.size()),one(ref.size());
  for(int t=0;t<8;t++)for(int r=0;r<m;r++){double z=0;for(int k=0;k<n;k++)z+=(double)deq[(size_t)r*n+k]*x[(size_t)t*n+k];ref[(size_t)t*m+r]=z;}
  for(int t=1;t<=8;t++){
   native_mmvq(sh.type,dw.p,dq.p,(float*)dy.p,n,m,t,s);
   for(int j=0;j<t;j++)native_mmvq(sh.type,dw.p,(char*)dq.p+j*native_q8_1_bytes(n,1),(float*)ds.p+j*m,n,m,1,s);
   ck(cudaStreamSynchronize(s));ck(cudaMemcpy(got.data(),dy.p,(size_t)t*m*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(one.data(),ds.p,(size_t)t*m*4,cudaMemcpyDeviceToHost));
   double err=0,norm=0;size_t diff=0;
   for(int k=0;k<t*m;k++){if(!std::isfinite(got[k]))throw std::runtime_error("nonfinite");double d=got[k]-ref[k];err+=d*d;norm+=(double)ref[k]*ref[k];if(memcmp(&got[k],&one[k],4))diff++;}
   if(dump.is_open()){dump.write(reinterpret_cast<const char*>(got.data()),(size_t)t*m*4);if(!dump)throw std::runtime_error("output write");}
   double rel=sqrt(err/norm);if(rel>.04||diff)throw std::runtime_error("numerical/column parity failed");
   cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
   for(int i=0;i<5;i++)native_mmvq(sh.type,dw.p,dq.p,(float*)dy.p,n,m,t,s);
   ck(cudaEventRecord(a,s));for(int i=0;i<100;i++)native_mmvq(sh.type,dw.p,dq.p,(float*)dy.p,n,m,t,s);ck(cudaEventRecord(b,s));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
   printf("type=%s n=%d m=%d T=%d rel_l2=%.7f bit_diff=%zu us=%.3f\n",tr->type_name,n,m,t,rel,diff,ms*10);fflush(stdout);ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
  }
 }
 ck(cudaStreamDestroy(s));puts("RDNA MMVQ screen passed");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}

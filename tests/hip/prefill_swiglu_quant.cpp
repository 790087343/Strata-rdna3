// Exact quantized-byte parity against the original separate launches, including padding.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/hip_compat/cuda_runtime.h"
#include "ggml.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace m=strata::prefill::mmq;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(){
 try{
 int count=0;
 for(int type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS})
 for(int rows:{1,7,64,511,4096})for(int cols:{128,640,1024})for(bool interleaved:{false,true}){
  std::vector<float>x((size_t)rows*cols*2);for(size_t i=0;i<x.size();i++)x[i]=(i<(size_t)cols*2?0.f:std::sin(float(i)*.13f)*8.f);
  float *gu,*h;void *a,*b;size_t bytes=m::q8_bytes(rows,cols);ck(cudaMalloc((void**)&gu,x.size()*4));ck(cudaMalloc((void**)&h,(size_t)rows*cols*4));ck(cudaMalloc(&a,bytes));ck(cudaMalloc(&b,bytes));
  ck(cudaMemcpy(gu,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMemset(a,0xa5,bytes));ck(cudaMemset(b,0xa5,bytes));
  setenv("STRATA_PF_SWIGLU_Q8_FUSED","0",1);m::swiglu_quantize(gu,h,a,type,rows,cols,interleaved,nullptr);
  setenv("STRATA_PF_SWIGLU_Q8_FUSED","1",1);m::swiglu_quantize(gu,h,b,type,rows,cols,interleaved,nullptr);
  ck(cudaDeviceSynchronize());std::vector<unsigned char>ha(bytes),hb(bytes);ck(cudaMemcpy(ha.data(),a,bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),b,bytes,cudaMemcpyDeviceToHost));
  if(ha!=hb){size_t i=0;while(ha[i]==hb[i])i++;std::fprintf(stderr,"FAIL type%d rows%d cols%d interleaved%d byte%zu %u/%u\n",type,rows,cols,interleaved,i,ha[i],hb[i]);return 1;}
  ck(cudaFree(gu));ck(cudaFree(h));ck(cudaFree(a));ck(cudaFree(b));count++;
 }
 std::printf("PASS %d SwiGLU+quantization cases, exact bytes including padded/zero rows\n",count);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}

// Mixed CPU/peer/hit rows, including direct remote scatter, with poisoned destinations.
#include <hip/hip_runtime.h>
#include "strata/kernels/elementwise.hpp"
#include <vector>
#include <cstdio>
#include <stdexcept>
#define HIP(x) do{auto check_error=(x);if(check_error!=hipSuccess)throw std::runtime_error(hipGetErrorString(check_error));}while(0)
int main(){try{
 for(int primary=0;primary<2;primary++)for(int rows:{10,40,80}){
  const int peer=1-primary,width=2560;HIP(hipSetDevice(primary));auto e=hipDeviceEnablePeerAccess(peer,0);if(e==hipErrorPeerAccessAlreadyEnabled)hipGetLastError();else HIP(e);
  float *host,*device_peer,*out;int32_t *kind,*hits,*count;
  HIP(hipHostMalloc((void**)&host,rows*width*4,hipHostMallocPortable|hipHostMallocMapped));HIP(hipHostMalloc((void**)&kind,rows*4,hipHostMallocPortable|hipHostMallocMapped));
  HIP(hipMalloc((void**)&device_peer,rows*width*4));HIP(hipMalloc((void**)&out,rows*width*4));HIP(hipMalloc((void**)&hits,rows*4));HIP(hipMalloc((void**)&count,4));
  std::vector<int> hit,remote;std::vector<float> compact;
  for(int r=0;r<rows;r++){kind[r]=r%4-1;for(int j=0;j<width;j++)host[r*width+j]=float(r*width+j);if(kind[r]==0||kind[r]==1)hit.push_back(r);if(kind[r]==2){remote.push_back(r);for(int j=0;j<width;j++)compact.push_back(-float(1+r*width+j));}}
  int nh=hit.size();HIP(hipMemcpy(hits,hit.data(),hit.size()*4,hipMemcpyHostToDevice));HIP(hipMemcpy(count,&nh,4,hipMemcpyHostToDevice));HIP(hipMemset(device_peer,0xff,rows*width*4));HIP(hipMemset(out,0xff,rows*width*4));HIP(hipDeviceSynchronize());
  HIP(hipSetDevice(peer));e=hipDeviceEnablePeerAccess(primary,0);if(e==hipErrorPeerAccessAlreadyEnabled)hipGetLastError();else HIP(e);
  float* src;int32_t* index;HIP(hipMalloc((void**)&src,compact.size()*4));HIP(hipMalloc((void**)&index,remote.size()*4));HIP(hipMemcpy(src,compact.data(),compact.size()*4,hipMemcpyHostToDevice));HIP(hipMemcpy(index,remote.data(),remote.size()*4,hipMemcpyHostToDevice));
  strata::kernels::scatter_rows_f32(src,device_peer,index,remote.size(),width,nullptr);HIP(hipDeviceSynchronize());
  HIP(hipSetDevice(primary));strata::kernels::copy_rows_from_mapped(out,host,rows,width,hits,count,nullptr,device_peer,kind);
  std::vector<float> actual(rows*width);HIP(hipMemcpy(actual.data(),out,actual.size()*4,hipMemcpyDeviceToHost));
  for(int r=0;r<rows;r++)for(int j=0;j<width;j++){float want=kind[r]==2?-float(1+r*width+j):kind[r]>=0?0.0f:host[r*width+j];if(actual[r*width+j]!=want){fprintf(stderr,"primary=%d rows=%d row=%d col=%d kind=%d actual=%g expected=%g\n",primary,rows,r,j,kind[r],actual[r*width+j],want);throw std::runtime_error("mixed peer row mismatch");}}
  HIP(hipFree(device_peer));HIP(hipFree(out));HIP(hipFree(hits));HIP(hipFree(count));HIP(hipHostFree(host));HIP(hipHostFree(kind));HIP(hipSetDevice(peer));HIP(hipFree(src));HIP(hipFree(index));
  printf("primary=%d rows=%d width=%d passed\n",primary,rows,width);
 }
 return 0;
}catch(const std::exception&e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}

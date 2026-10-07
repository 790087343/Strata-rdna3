// Experimental transport; included inside strata::core in peer_experts.cpp.
// No allocation in a layer. Primary transfer stream must never be the verifier's waiting graph stream.
class PeerTransport {
public:
    std::string mode;
    int peer;
    float* result_target=nullptr;
    size_t result_entries=0;
    bool check=std::getenv("STRATA_PEER_CHECK")!=nullptr;
    cudaStream_t primary_stream = nullptr;
    cudaEvent_t produced = nullptr, input_ready = nullptr;
    float* receive = nullptr;
    double launch_ms = 0, reported_launch_ms = 0, reported_wait_ms = 0;
    uint64_t calls = 0, input_bytes = 0, output_bytes = 0, last_calls = 0;
#if defined(STRATA_USE_HIP) && defined(__linux__)
    void* library = nullptr;
    ncclComm_t comm[2]{};
    decltype(&ncclCommInitAll) init = nullptr;
    decltype(&ncclCommDestroy) destroy = nullptr;
    decltype(&ncclGroupStart) group_start = nullptr;
    decltype(&ncclGroupEnd) group_end = nullptr;
    decltype(&ncclSend) send = nullptr;
    decltype(&ncclRecv) recv = nullptr;
    decltype(&ncclGetErrorString) error_string = nullptr;
    void nc(ncclResult_t e) { if(e!=ncclSuccess)throw std::runtime_error(std::string("RCCL: ")+error_string(e)); }
    template<class T> void symbol(T& f,const char* name) { f=reinterpret_cast<T>(dlsym(library,name));if(!f)throw std::runtime_error(std::string("RCCL missing symbol ")+name); }
#endif
    static void cu(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
    PeerTransport(std::string name,int dev):mode(std::move(name)),peer(dev) {}
    void open() {
        {On on(0);cu(cudaStreamCreateWithFlags(&primary_stream,cudaStreamNonBlocking));cu(cudaEventCreateWithFlags(&input_ready,cudaEventDisableTiming));cu(cudaMalloc((void**)&receive,(size_t)CAP*H*sizeof(float)));}
        {On on(peer);cu(cudaEventCreateWithFlags(&produced,cudaEventDisableTiming));}
        if(mode=="rccl") {
#if defined(STRATA_USE_HIP) && defined(__linux__)
            library=dlopen("librccl.so.1",RTLD_NOW|RTLD_LOCAL);
            if(!library)throw std::runtime_error(std::string("RCCL load: ")+dlerror());
            symbol(init,"ncclCommInitAll");symbol(destroy,"ncclCommDestroy");symbol(group_start,"ncclGroupStart");symbol(group_end,"ncclGroupEnd");symbol(send,"ncclSend");symbol(recv,"ncclRecv");symbol(error_string,"ncclGetErrorString");
            On on(0);int devices[]={0,peer};nc(init(comm,2,devices));
#else
            throw std::runtime_error("RCCL experiment requires Linux HIP build");
#endif
        }
        std::fprintf(stderr,"strata peer transport: %s, device activation input and primary-VRAM result rows\n",mode.c_str());
    }
    void input(float* dst,const float* src,const float* host,size_t bytes,cudaStream_t peer_stream) {
        input_bytes+=bytes;++calls;
        if(mode=="hoststore"){On on(peer);cu(cudaMemcpyAsync(dst,host,bytes,cudaMemcpyHostToDevice,peer_stream));return;}
        if(mode=="rccl") {
#if defined(STRATA_USE_HIP) && defined(__linux__)
            nc(group_start());{On on(0);nc(send(src,bytes,ncclUint8,1,comm[0],primary_stream));}{On on(peer);nc(recv(dst,bytes,ncclUint8,0,comm[1],peer_stream));}nc(group_end());
#endif
        }else if((mode=="push" || mode=="pushstore")) {
            {On on(0);strata::kernels::copy_from_mapped(dst,src,bytes/sizeof(float),primary_stream);cu(cudaEventRecord(input_ready,primary_stream));}
            {On on(peer);cu(cudaStreamWaitEvent(peer_stream,input_ready,0));}
        }else{On on(peer);cu(cudaMemcpyPeerAsync(dst,peer,src,0,bytes,peer_stream));}
    }
    void result(const float* src,size_t bytes,cudaStream_t peer_stream) {
        output_bytes+=bytes;
        if(mode=="rccl") {
#if defined(STRATA_USE_HIP) && defined(__linux__)
            nc(group_start());{On on(peer);nc(send(src,bytes,ncclUint8,0,comm[1],peer_stream));}{On on(0);nc(recv(receive,bytes,ncclUint8,1,comm[0],primary_stream));}nc(group_end());
#endif
        }else{
            {On on(peer);cu(cudaEventRecord(produced,peer_stream));}
            {On on(0);cu(cudaStreamWaitEvent(primary_stream,produced,0));cu(cudaMemcpyPeerAsync(receive,0,src,peer,bytes,primary_stream));}
        }
    }
    void report(double wait_ms) {
        std::fprintf(stderr,"strata peer transport timing: %s calls %llu launch_host_ms %.3f finish_wait_ms %.3f; cumulative input_bytes %llu output_bytes %llu\n",mode.c_str(),(unsigned long long)(calls-last_calls),launch_ms-reported_launch_ms,wait_ms-reported_wait_ms,(unsigned long long)input_bytes,(unsigned long long)output_bytes);
        last_calls=calls;reported_launch_ms=launch_ms;reported_wait_ms=wait_ms;
    }
    ~PeerTransport() {
        {On on(0);if(primary_stream)cudaStreamSynchronize(primary_stream);}
#if defined(STRATA_USE_HIP) && defined(__linux__)
        for(int i=0;i<2;i++)if(comm[i]){On on(i?peer:0);destroy(comm[i]);}
        if(library)dlclose(library);
#endif
        {On on(0);if(input_ready)cudaEventDestroy(input_ready);if(receive)cudaFree(receive);if(primary_stream)cudaStreamDestroy(primary_stream);}
        {On on(peer);if(produced)cudaEventDestroy(produced);}
    }
};

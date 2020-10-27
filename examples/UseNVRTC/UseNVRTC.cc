//  https://docs.nvidia.com/cuda/nvrtc/index.html#basic-usage

#include <exception>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <nvrtc.h>
#include <cuda.h>

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

#define NVRTC_CHECK_ERROR( func )                                  \
  do {                                                             \
    nvrtcResult code = func;                                       \
    if( code != NVRTC_SUCCESS )                                    \
      throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) ); \
  } while( 0 )

const char* cu_source = R"TOKEN(
extern "C" __global__                                         
void saxpy(float a, float *x, float *y, float *out, size_t n)  
{                                                              
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;          
  if (tid < n) {                                               
    out[tid] = a * x[tid] + y[tid];                            
  }                                                            
}                                                              
)TOKEN";


int main(int argc, char** argv)
{
    nvrtcProgram prog = 0;
    const char* name = "saxpy.cu" ; 
    int  numHeaders = 0 ; 
    const char** headers = NULL ; 
    const char** includeNames = NULL ; 
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, numHeaders, headers, includeNames )) ;

    const char* opts[] = {"--gpu-architecture=compute_30" } ;
    int numOptions = 1 ; 
    NVRTC_CHECK_ERROR( nvrtcCompileProgram(prog, numOptions, opts)) ;

    size_t logSize;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize(prog, &logSize) );

    char* log = new char[logSize];
    NVRTC_CHECK_ERROR( nvrtcGetProgramLog(prog, log) );

    size_t ptxSize;
    NVRTC_CHECK_ERROR( nvrtcGetPTXSize(prog, &ptxSize));

    char *ptx = new char[ptxSize];
    NVRTC_CHECK_ERROR( nvrtcGetPTX(prog, ptx) );

    NVRTC_CHECK_ERROR( nvrtcDestroyProgram(&prog) );

     
    std::cout 
        << "[log size " << logSize 
        << std::endl 
        << log 
        << std::endl 
        << "]log" 
        << std::endl
        ;
 
    std::cout 
        << "[ptx size " << ptxSize
        << std::endl 
        << ptx 
        << std::endl
        << "]ptx" 
        << std::endl
        ;



    const unsigned NUM_THREADS = 100 ; 
    const unsigned NUM_BLOCKS = 1 ; 

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);

    const char* function = "saxpy" ; 
    cuModuleGetFunction(&kernel, module, function);

    size_t n = NUM_THREADS * NUM_BLOCKS;
    size_t bufferSize = n * sizeof(float);

    std::vector<float> hX(n, 0.f);  
    std::vector<float> hY(n, 0.f);  
    std::vector<float> hOut(n, 0.f); 

    for(unsigned i=0 ; i < n ; i++)
    {
        hX[i] = float(i) ; 
        hY[i] = float(i) ; 
    } 

    std::cout << "----" << std::endl ;    

    float a = 10 ; 

    CUdeviceptr dX, dY, dOut;
    cuMemAlloc(&dX, bufferSize);
    cuMemAlloc(&dY, bufferSize);
    cuMemAlloc(&dOut, bufferSize);

    cuMemcpyHtoD(dX, hX.data(), bufferSize);
    cuMemcpyHtoD(dY, hY.data(), bufferSize);

    void *args[] = { &a, &dX, &dY, &dOut, &n };

    std::cout << "launch" << std::endl ; 

    cuLaunchKernel(kernel,
                   NUM_THREADS, 1, 1,   // grid dim
                   NUM_BLOCKS, 1, 1,    // block dim
                   0, NULL,             // shared mem and stream
                   args,                // arguments
                   0);
    cuCtxSynchronize();
    cuMemcpyDtoH(hOut.data(), dOut, bufferSize);

    for(unsigned i=0 ; i < n ; i++)
    {
        std::cout << std::setw(7) << hOut[i] << " " ; 
        if( i % 20 == 0 ) std::cout << std::endl ; 
    }
    std::cout << std::endl ; 


    return 0 ; 
}



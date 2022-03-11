//  https://docs.nvidia.com/cuda/nvrtc/index.html#basic-usage

#include <exception>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <string>
#include <vector>

#include "Prog.h"

#include <cuda.h>


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
    //  source string -> ptx string

    Prog* prog = nullptr ; 
    {
        const char* name = "saxpy.cu" ; 
        int  numHeaders = 0 ; 
        const char** headers = nullptr ; 
        const char** includeNames = nullptr ; 
        prog = new Prog(name, cu_source, numHeaders, headers, includeNames ); 
    }

    const char* opts[] = {"--gpu-architecture=compute_30" } ;
    int numOptions = 1 ; 
    prog->compile(numOptions, opts); 
    prog->dump();  

    const char* ptx = prog->ptx ; 



    // ptx string -> module 

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
    std::vector<float> hOutExpect(n, 0.f); 

    float a = 10 ; 

    for(unsigned i=0 ; i < n ; i++)
    {
        hX[i] = float(i) ; 
        hY[i] = float(i) ; 

        hOutExpect[i] = a*hX[i] + hY[i] ; 
    } 

    std::cout << "----" << std::endl ;    


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

    for(unsigned i=0 ; i < n ; i++) assert( hOut[i] == hOutExpect[i] ); 

    return 0 ; 
}



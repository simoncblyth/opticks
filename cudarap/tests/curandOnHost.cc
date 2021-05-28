// name=curandOnHost ; gcc $name.cc -std=c++11 -I/usr/local/cuda/include -L/usr/local/cuda/lib -lstdc++ -lcurand -lcudart -o /tmp/$name && /tmp/$name


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>


template<typename T>
struct curandOnHost
{
    unsigned n ; 
    unsigned long long seed ; 
    T* data ; 
    T* d_data ; 
    curandGenerator_t gen ; 

    curandOnHost(unsigned n_, unsigned long long seed_);
    ~curandOnHost(); 

    void init(); 
    void generate(); 
    void copy(); 
    void dump(); 
}; 


template<typename T>
curandOnHost<T>::curandOnHost(unsigned n_, unsigned long long seed_)
    :
    n(n_),
    seed(seed_),
    data((T*)calloc(n, sizeof(T))),
    d_data(nullptr)
{
    init(); 
}

template<typename T>
void curandOnHost<T>::init()
{
    cudaMalloc( (void**)&d_data, n*sizeof(T)) ; 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32 ); 
    curandSetPseudoRandomGeneratorSeed(gen, seed ); 
}

template<typename T>
curandOnHost<T>::~curandOnHost()
{
    curandDestroyGenerator(gen); 
    cudaFree(d_data); 
    free(data);  
}

template<> void curandOnHost<float>::generate()
{
    curandGenerateUniform(gen, d_data, n); 
}

template<> void curandOnHost<double>::generate()
{
    curandGenerateUniformDouble(gen, d_data, n); 
}

template<typename T>
void curandOnHost<T>::copy()
{
    cudaMemcpy(data, d_data, n*sizeof(T), cudaMemcpyDeviceToHost ); 
}

template<typename T>
void curandOnHost<T>::dump()
{
    std::cout << "curandOnHost<T>::dump :  sizeof(T) " << sizeof(T) << std::endl ; 
    for(unsigned i=0 ; i < n ; i++) 
        std::cout << std::fixed << std::setw(16) << std::setprecision(15) << data[i] << " " << std::endl ; 
}




int main(int argc, char** argv)
{
    unsigned n = 10 ; 
    unsigned long long seed = 42ull ; 

    curandOnHost<float> genf(n, seed); 
    genf.generate(); 
    genf.copy(); 
    genf.dump();  

    curandOnHost<double> gend(n, seed); 
    gend.generate(); 
    gend.copy(); 
    gend.dump();  

    return 0 ; 
}


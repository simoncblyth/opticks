#include "QUDA_CHECK.h"

#include "QRng.hh"
#include "QGen.hh"
#include "PLOG.hh"

const plog::Severity QGen::LEVEL = PLOG::EnvLevel("QGen", "INFO"); 

QGen::QGen()
    :
    rng(QRng::Get())
{
}

extern "C" void QGen_generate_kernel(dim3 numBlocks, dim3 threadsPerBlock, curandState* d_rng_states, float* d_gen, unsigned num_gen ) ; 


void QGen::generate(float* dst, unsigned num_gen)
{
    long rngmax = rng ? rng->rngmax : 0 ;  
    LOG(LEVEL) 
        << "["
        << " rngmax " << rngmax
        << " num_gen " << num_gen
        ; 
    assert( num_gen < rngmax ); 


    float* d_gen ;  

    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_gen ), num_gen*sizeof(float) )); 

    unsigned width = num_gen ; 
    unsigned height = 1 ; 

    dim3 threadsPerBlock(512, 1);

    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    QGen_generate_kernel(numBlocks, threadsPerBlock, rng->d_rng_states, d_gen, num_gen );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( dst ), d_gen, sizeof(float)*num_gen, cudaMemcpyDeviceToHost )); 

    QUDA_CHECK( cudaFree(d_gen) ); 

    LOG(LEVEL) << "]" ; 
}

void QGen::dump(float* dst, unsigned num_gen)
{
    LOG(LEVEL) << "[" ; 
    if( dst == nullptr ) return ; 
    for(unsigned i=0 ; i < num_gen ; i++ ) 
    {
        std::cout 
            << std::setw(4) << i 
            << " : "
            << std::setw(10) << std::fixed << std::setprecision(7) << dst[i] 
            << std::endl
            ; 
    }
    LOG(LEVEL) << "]" ; 
}


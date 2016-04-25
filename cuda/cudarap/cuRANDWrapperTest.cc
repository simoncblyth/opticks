#include "LaunchCommon.hh"
#include "LaunchSequence.hh"
#include "cuRANDWrapper.hh"

//#define WORK 1024*768
#define WORK 1024*1


int main(int argc, char** argv)
{
    unsigned int work              = getenvvar("CUDARAP_RNG_MAX", WORK) ;
    unsigned long long seed        = 0 ;
    unsigned long long offset      = 0 ;
    unsigned int max_blocks        = getenvvar("MAX_BLOCKS", 128) ;
    unsigned int threads_per_block = getenvvar("THREADS_PER_BLOCK", 256) ; 

    char* cachedir = getenv("CUDARAP_RNG_DIR") ;

    cuRANDWrapper* crw = cuRANDWrapper::instanciate( work, cachedir, seed, offset, max_blocks, threads_per_block );

    crw->Allocate();
    crw->InitFromCacheIfPossible(); 
    // CAUTION: without Init still provides random numbers but different ones every time

    // can increase max_blocks as generation much faster than initialization 
    crw->getLaunchSequence()->setMaxBlocks(max_blocks*32);  

    crw->Test();

    crw->Summary("cuRANDWrapperTest::main");



    crw->resize( 1024*10 );

    crw->Test();

    crw->Summary("cuRANDWrapperTest::main after resize");



}


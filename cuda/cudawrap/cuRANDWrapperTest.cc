#include "LaunchCommon.hh"
#include "LaunchSequence.hh"
#include "cuRANDWrapper.hh"

//#define WORK 1024*768
#define WORK 1024*1


int main(int argc, char** argv)
{
    unsigned int work              = getenvvar("WORK", WORK) ;
    unsigned int threads_per_block = getenvvar("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks        = getenvvar("MAX_BLOCKS", 128) ;
    bool reverse                   = false ; 

    LaunchSequence* seq = new LaunchSequence( work, threads_per_block, max_blocks, reverse) ;

    cuRANDWrapper*  crw = new cuRANDWrapper(seq);

    crw->setCacheDir("/tmp/env/cuRANDWrapperTest/cachedir");

    bool create = true ; 
    crw->Setup(create);

    // can increase max_blocks as generation much faster than initialization 
    crw->getLaunchSequence()->setMaxBlocks(max_blocks*32);  

    crw->Test();

    crw->Summary("cuRANDWrapperTest::main");

}


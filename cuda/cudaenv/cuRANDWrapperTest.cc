#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"
#include "LaunchCommon.hh"
#include "curand_kernel.h"


int main(int argc, char** argv)
{
    unsigned int work = getenvvar("WORK", 1024*768) ;
    unsigned int threads_per_block = getenvvar("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks = getenvvar("MAX_BLOCKS", 128) ;

    LaunchSequence* seq = new LaunchSequence( work, threads_per_block, max_blocks) ;
    seq->Summary("seq");

    cuRANDWrapper* crw = new cuRANDWrapper(seq);

    crw->create_rng();
    crw->init_rng();

    crw->test_rng();
    crw->test_rng();
    crw->test_rng();
    crw->test_rng();
    crw->test_rng();


}


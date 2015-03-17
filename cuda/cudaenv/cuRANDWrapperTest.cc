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
    bool reverse = true ; 

    LaunchSequence* seq = new LaunchSequence( work, threads_per_block, max_blocks, reverse) ;
    seq->Summary("seq");

    cuRANDWrapper* crw = new cuRANDWrapper(seq);

    crw->create_rng();

    crw->init_rng("init");


    LaunchSequence* tseq = seq->copy(max_blocks*32); // can increase max_blocks as test_rng much faster than init_rng 
    crw->setLaunchSequence(tseq);

    crw->test_rng("test_0");
    crw->test_rng("test_1");
    crw->test_rng("test_2");
    crw->test_rng("test_3");
    crw->test_rng("test_4");

    crw->Summary("crw");


}


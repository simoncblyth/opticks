#include "LaunchSequence.hh"
#include "LaunchCommon.hh"

int main(int argc, char** argv)
{
    unsigned int work = getenvvar("WORK", 1024*768) ;
    unsigned int threads_per_block = getenvvar("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks = getenvvar("MAX_BLOCKS", 128) ;

    LaunchSequence seq( work, threads_per_block, max_blocks) ;
    seq.Summary("seq");
}


#include "LaunchSequence.hh"
#include "LaunchCommon.hh"

#include "PLOG.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    unsigned int work = SSys::getenvint("WORK", 1024*768) ;
    unsigned int threads_per_block = SSys::getenvint("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks = SSys::getenvint("MAX_BLOCKS", 128) ;

    LOG(info) << argv[0]
              << " work " << work 
              << " threads_per_block " << threads_per_block 
              << " max_blocks " << max_blocks
              ; 

    LaunchSequence seq( work, threads_per_block, max_blocks) ;
    seq.Summary("seq");
}


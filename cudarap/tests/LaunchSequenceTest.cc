#include "LaunchSequence.hh"
#include "LaunchCommon.hh"

#include "OPTICKS_LOG.hh"
#include "SSys.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned int work = SSys::getenvint("WORK", 1024*768) ;
    unsigned int threads_per_block = SSys::getenvint("THREADS_PER_BLOCK", 256) ; 
    unsigned int max_blocks = SSys::getenvint("MAX_BLOCKS", 128) ;

    LOG(info) << argv[0]
              << " work " << work 
              << " threads_per_block " << threads_per_block 
              << " max_blocks " << max_blocks
              ; 

    LaunchSequence seq( work, threads_per_block, max_blocks) ;
    seq.dump("seq");
}

/*

simon:cudarap blyth$ LaunchSequenceTest
2017-12-02 14:02:48.277 INFO  [1088456] [main@15] LaunchSequenceTest work 786432 threads_per_block 256 max_blocks 128
seq tag  workitems  786432  threads_per_block   256  max_blocks    128 reverse 0 nlaunch  24 TotalTime     0.0000 ms 
 seq sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   1  thread_offset   32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   2  thread_offset   65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   3  thread_offset   98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   4  thread_offset  131072  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   5  thread_offset  163840  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   6  thread_offset  196608  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   7  thread_offset  229376  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   8  thread_offset  262144  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index   9  thread_offset  294912  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  10  thread_offset  327680  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  11  thread_offset  360448  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  12  thread_offset  393216  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  13  thread_offset  425984  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  14  thread_offset  458752  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  15  thread_offset  491520  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  16  thread_offset  524288  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  17  thread_offset  557056  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  18  thread_offset  589824  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  19  thread_offset  622592  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  20  thread_offset  655360  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  21  thread_offset  688128  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  22  thread_offset  720896  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
 seq sequence_index  23  thread_offset  753664  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    -1.0000 ms 
simon:cudarap blyth$ 

*/




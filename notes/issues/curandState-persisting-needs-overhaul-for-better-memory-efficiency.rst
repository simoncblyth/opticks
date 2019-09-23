curandState-persisting-needs-overhaul-for-better-memory-efficiency
===================================================================

Context
--------

* :doc:`gpu-photon-generation-needed-to-push-ceiling`


Issue
--------

::

    239 scan-rngmax-opt-notes(){ cat << EON
    240 
    241 * this simple approach of a range of fixed sizes means
    242   that will almost always be using a lot more memory 
    243   for rng_states than is necessary 
    244 
    245 * better to use the building block approach with each block 
    246   corresponding to 10M slots 
    247 
    248 
    249 EON
    250 }
    251 
    252 
    253 scan-rngmax-opt(){ 
    254    local num_photons=${1:-0}
    255    
    256    local M=$(( 1000000 ))
    257    local M3=$(( 3*M ))
    258    local M10=$(( 10*M ))
    259    local M100=$(( 100*M ))
    260    local M200=$(( 200*M ))
    261    local M400=$(( 400*M ))
    262    
    263    local opt
    264    
    265    if [ $num_photons -gt $M400 ]; then
    266       echo $msg num_photons $num_photons is above the ceiling 
    267    elif [ $num_photons -gt $M200 ]; then
    268        opt="--rngmax 400"
    269    elif [ $num_photons -gt $M100 ]; then
    270        opt="--rngmax 200"
    271    elif [ $num_photons -gt $M10 ]; then
    272        opt="--rngmax 100"
    273    elif [ $num_photons -gt $M3 ]; then
    274        opt="--rngmax 10"
    275    else
    276        opt="--rngmax 3"
    277    fi
    278    echo $opt
    279 }
    280 



* 100M takes 4.1G better to organize into files of 25M about 1G each 


::

    [blyth@localhost cu]$ cudarap-rngdir-du
    4.1G    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_100000000_0_0.bin
    420M    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10000000_0_0.bin
    42M /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin
    440K    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10240_0_0.bin
    8.2G    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_200000000_0_0.bin
    84M /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_2000000_0_0.bin
    126M    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin
    [blyth@localhost cu]$ 



Silver 400M::

      
     ...
     init_rng_wrapper sequence_index 12200  thread_offset 399769600  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   193.7314 ms 
     init_rng_wrapper sequence_index 12201  thread_offset 399802368  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   194.3307 ms 
     init_rng_wrapper sequence_index 12202  thread_offset 399835136  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   194.8300 ms 
     init_rng_wrapper sequence_index 12203  thread_offset 399867904  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   195.3678 ms 
     init_rng_wrapper sequence_index 12204  thread_offset 399900672  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   195.9245 ms 
     init_rng_wrapper sequence_index 12205  thread_offset 399933440  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   196.4776 ms 
     init_rng_wrapper sequence_index 12206  thread_offset 399966208  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   197.0076 ms 
     init_rng_wrapper sequence_index 12207  thread_offset 399998976  threads_per_launch   1024 blocks_per_launch      4   threads_per_block    256  kernel_time    21.9009 ms 
     init_rng_wrapper tag init workitems 400000000  threads_per_block   256  max_blocks    128 reverse 0 nlaunch 12208 TotalTime 2230874.2500 ms 

::

    (base) [blyth@gilda03 ~]$ cudarap-rngdir-du
    4.1G    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_100000000_0_0.bin
    420M    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10000000_0_0.bin
    42M /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin
    440K    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_10240_0_0.bin
    8.2G    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_200000000_0_0.bin
    126M    /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin
    17G /home/blyth/.opticks/rngcache/RNG/cuRANDWrapper_400000000_0_0.bin
    (base) [blyth@gilda03 ~]$ 





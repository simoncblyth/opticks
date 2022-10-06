opticks-prepare-installation-needs-updating-from-cudarap-to-QUDARap-binary
==============================================================================


Issue Reported by Ami
-----------------------

> Unfortunately, at the end of the new Opticks installation, I get the following errors:
> == opticks-full-make : DONE Wed 05 Oct 2022 11:48:04 PM EDT 
> === opticks-full : detected GPU proceed with opticks-full-prepare
> === opticks-full-prepare : START Wed 05 Oct 2022 11:48:04 PM EDT 
> === opticks-prepare-installation : generating RNG seeds into installcache
> bash: /home/ami/Documents/codes/new_opticks/test/local/opticks/lib/cuRANDWrapperTest: No such file or directory
> bash: /home/ami/Documents/codes/new_opticks/test/local/opticks/lib/cuRANDWrapperTest: No such file or directory
> bash: /home/ami/Documents/codes/new_opticks/test/local/opticks/lib/cuRANDWrapperTest: No such file or directory
> I recall this error occurred for the old opticks and it was related to the 
> cuda-10.1--as I used cuda-10.1 I never got the error. But I'm currently using
> cuda-10.1 as it is shown below > 


This issue is that the installation is trying to use a binary from the 
old opticks cudarap package which is no longer part of the standard build. 
I need to update the installation to do the curand_init RNG "seeding" preparations 
with a qudarap based binary.   


Compare QSim/QRng setup with cudarap, identify new binary, check installation paths
------------------------------------------------------------------------------------------

::

     101 void QSim::UploadComponents( const SSim* ssim  )
     102 {
     106     QBase* base = new QBase ;
     ...
     113     QRng* rng = new QRng ;  // loads and uploads curandState 


QRng instanciation does indeed load from DEFAULT_PATH and upload to device::

     15 const char* QRng::DEFAULT_PATH = SPath::Resolve("$HOME/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin", 0) ;
     16 //const char* QRng::DEFAULT_PATH = SPath::Resolve("$HOME/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin", 0) ; 
     17 
     18 QRng::QRng(const char* path_, unsigned skipahead_event_offset)
     19     :
     20     path(path_ ? strdup(path_) : DEFAULT_PATH),
     21     rngmax(0),
     22     rng_states(Load(rngmax, path)),
     23     qr(new qrng(skipahead_event_offset)),
     24     d_qr(nullptr)
     25 {   
     26     INSTANCE = this ;
     27     upload();
     28 }
     29 


HMM: Looks like QRng does not have kernels to generate the curandState files yet 
and is using files generated from the old cudarap binary.

cudarap workflow for generating curandState files
----------------------------------------------------

Look at old workflow::

    epsilon:tests blyth$ t cudarap-prepare-installation
    cudarap-prepare-installation () 
    { 
        local size;
        cudarap-prepare-sizes | while read size; do
            CUDARAP_RNGMAX_M=$size cudarap-prepare-rng-;
        done
    }

    epsilon:tests blyth$ t cudarap-prepare-sizes-Linux-
    cudarap-prepare-sizes-Linux- () 
    { 
        echo ${OPTICKS_CUDARAP_RNGMAX:-1,3,10}
    }
    epsilon:tests blyth$ t cudarap-prepare-sizes-Darwin-
    cudarap-prepare-sizes-Darwin- () 
    { 
        echo ${OPTICKS_CUDARAP_RNGMAX:-1,3}
    }

    epsilon:tests blyth$ t cudarap-rngmax-M
    cudarap-rngmax-M () 
    { 
        echo ${CUDARAP_RNGMAX_M:-3}
    }

    epsilon:tests blyth$ t cudarap-prepare-rng-
    cudarap-prepare-rng- () 
    { 
        local msg="=== $FUNCNAME :";
        local path=$(cudarap-rngpath);
        [ -f "$path" ] && echo $msg path $path exists already && return 0;
        CUDARAP_RNG_DIR=$(cudarap-rngdir) CUDARAP_RNG_MAX=$(cudarap-rngmax) $(cudarap-ibin)
    }

    epsilon:tests blyth$ cudarap-rngdir
    /Users/blyth/.opticks/rngcache/RNG

    epsilon:~ blyth$ cudarap-ibin
    /usr/local/opticks/lib/cuRANDWrapperTest


    epsilon:tests blyth$ t cudarap-rngmax
    cudarap-rngmax () 
    { 
        echo $(( $(cudarap-rngmax-M)*1000*1000 ))
    }



    epsilon:tests blyth$ cudarap-prepare-installation
    === cudarap-prepare-rng- : path /Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin exists already
    === cudarap-prepare-rng- : path /Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_3000000_0_0.bin exists already

cudarap/tests/cuRANDWrapperTest.cc::

     44 int main(int argc, char** argv)
     45 {
     46     OPTICKS_LOG(argc, argv);
     47 
     48     unsigned int work              = SSys::getenvint("CUDARAP_RNG_MAX", WORK) ;
     49     unsigned long long seed        = 0 ;
     50     unsigned long long offset      = 0 ;
     51     unsigned int max_blocks        = SSys::getenvint("MAX_BLOCKS", 128) ;
     52     unsigned int threads_per_block = SSys::getenvint("THREADS_PER_BLOCK", 256) ;
     53 
     54     int create_dirs = 2 ; // 2:directory path argument
     55     const char* tmp = SPath::Resolve("$TMP", create_dirs );
     56     const char* cachedir = SSys::getenvvar("CUDARAP_RNG_DIR", tmp) ;
     57 
     58     LOG(info)
     59           << " work " << work
     60           << " max_blocks " << max_blocks
     61           << " seed " << seed
     62           << " offset " << offset
     63           << " threads_per_block " << threads_per_block
     64           << " cachedir " << cachedir
     65           ;
     66 
     67 
     68     cuRANDWrapper* crw = cuRANDWrapper::instanciate( work, cachedir, seed, offset, max_blocks, threads_per_block );
     69 
     70     crw->Allocate();
     71     crw->InitFromCacheIfPossible();
     72     // CAUTION: without Init still provides random numbers but different ones every time
     73 
     74     // can increase max_blocks as generation much faster than initialization 
     75 
     76     const LaunchSequence* launchseq = crw->getLaunchSequence() ;
     77     const_cast<LaunchSequence*>(launchseq)->setMaxBlocks(max_blocks*32);
     78 
     79     crw->Test();
     80 
     81     crw->Summary("cuRANDWrapperTest::main");


Hmm these are real ancient (first non-trivial CUDA) : so they are a mess.
Needs a rethink to do more simply and more modular::

    epsilon:cudarap blyth$ vi cuRANDWrapper.hh cuRANDWrapper_kernel.cu cuRANDWrapper.cc cuRANDWrapper_kernel.hh


Starting in qudarap/QCurandState.hh


::

    epsilon:qudarap blyth$ ls -l /tmp/QCurandState.bin
    -rw-r--r--  1 blyth  wheel  44000000 Oct  6 17:13 /tmp/QCurandState.bin


    epsilon:qudarap blyth$ ls -l /tmp/QCurandState.bin
    -rw-r--r--  1 blyth  wheel  44000000 Oct  6 17:13 /tmp/QCurandState.bin
    epsilon:qudarap blyth$ ls -l /Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin
    -rw-r--r--  1 blyth  staff  44000000 Apr  6  2020 /Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin
    epsilon:qudarap blyth$ diff -b /tmp/QCurandState.bin /Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin
    epsilon:qudarap blyth$ 
    epsilon:qudarap blyth$ rc
    RC 0
    epsilon:qudarap blyth$ 



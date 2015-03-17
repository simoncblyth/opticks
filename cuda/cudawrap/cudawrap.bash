# === func-gen- : cuda/cudawrap/cudawrap fgp cuda/cudawrap/cudawrap.bash fgn cudawrap fgh cuda/cudawrap
cudawrap-src(){      echo cuda/cudawrap/cudawrap.bash ; }
cudawrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cudawrap-src)} ; }
cudawrap-vi(){       vi $(cudawrap-source) ; }
cudawrap-env(){      
  elocal-  
  cuda-
}
cudawrap-usage(){ cat << EOU

Warning::

    delta:2015 blyth$ cudawrap-make
    [ 16%] Building NVCC (Device) object CMakeFiles/CUDAEnv.dir//./CUDAEnv_generated_cuRANDWrapper_kernel.cu.o
    ptxas /tmp/tmpxft_00004fb0_00000000-5_cuRANDWrapper_kernel.ptx, line 894; warning : Double is not supported. Demoting to float
    Scanning dependencies of target CUDAEnv

    Created ptx has f64 on the offending line 894
    nvcc -ptx cuRANDWrapper_kernel.cu

    861     .entry _Z8test_rngiiP17curandStateXORWOWPf (
    862         .param .s32 __cudaparm__Z8test_rngiiP17curandStateXORWOWPf_threads_per_launch,
    863         .param .s32 __cudaparm__Z8test_rngiiP17curandStateXORWOWPf_thread_offset,
    864         .param .u64 __cudaparm__Z8test_rngiiP17curandStateXORWOWPf_rng_states,
    865         .param .u64 __cudaparm__Z8test_rngiiP17curandStateXORWOWPf_a)
    866     {
    ...
    887     ld.param.u64    %rd2, [__cudaparm__Z8test_rngiiP17curandStateXORWOWPf_rng_states];
    888     mul.wide.s32    %rd3, %r3, 48;
    889     add.u64     %rd4, %rd2, %rd3;
    890     ld.global.u32   %r5, [%rd4+4];
    891     ld.global.u32   %r6, [%rd4+20];
    892     ld.global.v2.s32    {%r7,%r8}, [%rd4+24];
    893     ld.global.f32   %f1, [%rd4+32];
    894     ld.global.f64   %fd1, [%rd4+40];
    895     .loc    14  125 0


Commenting rng on two lines in the below gets rid of the warning::

    112 __global__ void test_rng(int threads_per_launch, int thread_offset, curandState* rng_states, float *a)
    113 {
    114    //
    115 
    116     int id = blockIdx.x*blockDim.x + threadIdx.x;
    117     if (id >= threads_per_launch) return;
    118 
    119     // NB no id offsetting on rng_states or a, as the offsetting
    120     // was done once in the kernel call 
    121     // this means thread_offset argument not used
    122 
    123     curandState rng = rng_states[id];   
    124 
    125     //a[id] = curand_uniform(&rng);
    126 
    127     //rng_states[id] = rng;   
    128 }


From mdfind curand_kernel.h, /Developer/NVIDIA/CUDA-5.5/include/curand_kernel.h 
see that copying boxmuller_extra_double is causing the warning:: 

     121  * Implementation details not in reference documentation */
     122 struct curandStateXORWOW {
     123     unsigned int d, v[5];
     124     int boxmuller_flag;
     125     int boxmuller_flag_double;
     126     float boxmuller_extra;
     127     double boxmuller_extra_double;
     128 };
     ...
     ...       6*6 + 4 + 8 = 48 
     ...
     275 /*
     276  * Default RNG
     277  */
     278 /** \cond UNHIDE_TYPEDEFS */
     279 typedef struct curandStateXORWOW curandState_t;
     280 typedef struct curandStateXORWOW curandState;
     281 /** \endcond */



    [ 16%] Building NVCC (Device) object CMakeFiles/CUDAEnv.dir//./CUDAEnv_generated_cuRANDWrapper_kernel.cu.o
    /Users/blyth/env/cuda/cudawrap/cuRANDWrapper_kernel.cu(123): warning: variable "rng" was declared but never referenced

    /Users/blyth/env/cuda/cudawrap/cuRANDWrapper_kernel.cu(123): warning: variable "rng" was declared but never referenced


* http://stackoverflow.com/questions/19034321/cuda-double-demoted-to-float-and-understanding-ptx-output

* http://lists.tiker.net/pipermail/pycuda/2011-December/003513.html



delta:cudawrap blyth$ DYLD_LIBRARY_PATH=. WORK=$(( 1024*128 )) ./cuRANDWrapperTest 
seq workitems  131072  threads_per_block   256  max_blocks    128 nlaunch   4 
 seq sequence_index   0  thread_offset      0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   1  thread_offset  32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   2  thread_offset  65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   3  thread_offset  98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   0  thread_offset      0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   1  thread_offset  32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   2  thread_offset  65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   3  thread_offset  98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
delta:cudawrap blyth$ 



Comparing LaunchSequenceTest with cuda_launch.py 
Need to define all the envvars to get a match, as defaults not aligned::

    delta:cudawrap blyth$ ITEMS=$(( 1024*768 )) THREADS_PER_BLOCK=256 MAX_BLOCKS=256 cudawrap-lst
    seq workitems  786432  threads_per_block   256  max_blocks    256 nlaunch  12 
     seq sequence_index   0  thread_offset      0  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   1  thread_offset  65536  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   2  thread_offset 131072  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   3  thread_offset 196608  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   4  thread_offset 262144  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   5  thread_offset 327680  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   6  thread_offset 393216  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   7  thread_offset 458752  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   8  thread_offset 524288  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index   9  thread_offset 589824  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index  10  thread_offset 655360  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
     seq sequence_index  11  thread_offset 720896  threads_per_launch  65536 blocks_per_launch    256   threads_per_block    256  
    Launch1D work [786432] total 786432 max_blocks 256 threads_per_block 256 block (256, 1, 1) 
    offset          0 count 65536 grid (256, 1) block (256, 1, 1) 
    offset      65536 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     131072 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     196608 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     262144 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     327680 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     393216 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     458752 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     524288 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     589824 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     655360 count 65536 grid (256, 1) block (256, 1, 1) 
    offset     720896 count 65536 grid (256, 1) block (256, 1, 1) 
    delta:cudawrap blyth$ 


EOU
}

cudawrap-name(){ echo CUDAEnv ; }
cudawrap-bdir(){ echo $(local-base)/env/cuda/cudawrap.build ; }
cudawrap-idir(){ echo $(local-base)/env/cuda/cudawrap ; }
cudawrap-sdir(){ echo $(env-home)/cuda/cudawrap ; }


cudawrap-scd(){  cd $(cudawrap-sdir); }
cudawrap-bcd(){  cd $(cudawrap-bdir); }


cudawrap-wipe(){
   local msg=" === $FUNCNAME :"
   echo $msg 
   local bdir=$(cudawrap-bdir)
   rm -rf $bdir
}


cudawrap-cmake(){
   local iwd=$PWD

   [ -n "$WIPE" ] && cudawrap-wipe

   local bdir=$(cudawrap-bdir)
   mkdir -p $bdir

   cudawrap-bcd
 
   cmake -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_INSTALL_PREFIX=$(cudawrap-idir) \
         -DCUDA_NVCC_FLAGS="$(cuda-nvcc-flags)" \
          $(cudawrap-sdir) 

   cd $iwd
}

cudawrap-make(){
   local iwd=$PWD
   cudawrap-bcd
   local rc

   make $*
   rc=$?

   cd $iwd
   [ $rc -ne 0 ] && echo $FUNCNAME ERROR && return 1
   return 0
}


cudawrap-bin(){ echo $(cudawrap-bdir)/$(cudawrap-name)Test ; }
cudawrap-lst-bin(){ echo $(cudawrap-bdir)/LaunchSequenceTest ; }

cudawrap-lst(){ 
   local bin=$(cudawrap-lst-bin)
   $bin $*

   cuda_launch.py ${ITEMS:-$(( 1024*768 ))}
}


cudawrap-export(){
  echo -n
}

cudawrap-run(){
  if [ -n "$DEBUG" ]; then
      $DEBUG $(cudawrap-bin) -- $*
  else
      $(cudawrap-bin) $*
  fi
}


cudawrap--(){
  cudawrap-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1
  cudawrap-install $*
}

cudawrap-install(){
  cudawrap-make install 
}

cudawrap-test()
{
   local iwd=$PWD
   cudawrap-bcd
   DYLD_LIBRARY_PATH=. WORK=$(( 1024*768 )) ./cuRANDWrapperTest 
   cd $iwd
}




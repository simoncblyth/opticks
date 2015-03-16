# === func-gen- : cuda/cudaenv/cudaenv fgp cuda/cudaenv/cudaenv.bash fgn cudaenv fgh cuda/cudaenv
cudaenv-src(){      echo cuda/cudaenv/cudaenv.bash ; }
cudaenv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cudaenv-src)} ; }
cudaenv-vi(){       vi $(cudaenv-source) ; }
cudaenv-env(){      
  elocal-  
  cuda-
}
cudaenv-usage(){ cat << EOU



delta:cudaenv blyth$ DYLD_LIBRARY_PATH=. WORK=$(( 1024*128 )) ./cuRANDWrapperTest 
seq workitems  131072  threads_per_block   256  max_blocks    128 nlaunch   4 
 seq sequence_index   0  thread_offset      0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   1  thread_offset  32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   2  thread_offset  65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 seq sequence_index   3  thread_offset  98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   0  thread_offset      0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   1  thread_offset  32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   2  thread_offset  65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
 init_rng_wrapper sequence_index   3  thread_offset  98304  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  
delta:cudaenv blyth$ 



Comparing LaunchSequenceTest with cuda_launch.py 
Need to define all the envvars to get a match, as defaults not aligned::

    delta:cudaenv blyth$ ITEMS=$(( 1024*768 )) THREADS_PER_BLOCK=256 MAX_BLOCKS=256 cudaenv-lst
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
    delta:cudaenv blyth$ 


EOU
}

cudaenv-name(){ echo CUDAEnv ; }
cudaenv-bdir(){ echo $(local-base)/env/cuda/cudaenv ; }
cudaenv-sdir(){ echo $(env-home)/cuda/cudaenv ; }
cudaenv-scd(){  cd $(cudaenv-sdir); }
cudaenv-bcd(){  cd $(cudaenv-bdir); }


cudaenv-wipe(){
   local msg=" === $FUNCNAME :"
   echo $msg 
   local bdir=$(cudaenv-bdir)
   rm -rf $bdir
}


cudaenv-cmake(){
   local iwd=$PWD

   [ -n "$WIPE" ] && cudaenv-wipe

   local bdir=$(cudaenv-bdir)
   mkdir -p $bdir

   cudaenv-bcd
 
   cmake -DCMAKE_BUILD_TYPE=Debug \
         -DCUDA_NVCC_FLAGS="$(cuda-nvcc-flags)" \
          $(cudaenv-sdir) 

   cd $iwd
}

cudaenv-make(){
   local iwd=$PWD
   cudaenv-bcd
   local rc

   make $*
   rc=$?

   cd $iwd
   [ $rc -ne 0 ] && echo $FUNCNAME ERROR && return 1
   return 0
}


cudaenv-bin(){ echo $(cudaenv-bdir)/$(cudaenv-name)Test ; }
cudaenv-lst-bin(){ echo $(cudaenv-bdir)/LaunchSequenceTest ; }

cudaenv-lst(){ 
   local bin=$(cudaenv-lst-bin)
   $bin $*

   cuda_launch.py ${ITEMS:-$(( 1024*768 ))}
}


cudaenv-export(){
  echo -n
}

cudaenv-run(){
  if [ -n "$DEBUG" ]; then
      $DEBUG $(cudaenv-bin) -- $*
  else
      $(cudaenv-bin) $*
  fi
}


cudaenv--(){
  cudaenv-make
  [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1
  cudaenv-export 
  cudaenv-run $*
}



cudaenv-test()
{
   local iwd=$PWD
   cudaenv-bcd
   DYLD_LIBRARY_PATH=. WORK=$(( 1024*768 )) ./cuRANDWrapperTest 
   cd $iwd
}


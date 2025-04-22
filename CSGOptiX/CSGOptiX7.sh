#!/bin/sh
usage(){ cat << EOU

   source CSGOptiX7.sh



/usr/local/cuda-12.4/bin/nvcc 
     -forward-unknown-to-host-compiler 
     -DCONFIG_Debug 
     -DDEBUG_PIDX 
     -DDEBUG_PIDXYZ 
     -DOPTICKS_CSGOPTIX 
     -DRNG_PHILOX 
     -DWITH_PRD 
     -DWITH_RENDER 
     -DWITH_SIMTRACE 
     -DWITH_SIMULATE 
     -DWITH_THRUST 
     --options-file CMakeFiles/CSGOptiXPTX.dir/includes_CUDA.rsp 
     -g 
     -std=c++17 
     --generate-code=arch=compute_70,code=[compute_70,sm_70] 
     --use_fast_math 
     -MD      
     -MT CMakeFiles/CSGOptiXPTX.dir/CSGOptiX7.ptx 
     -MF CMakeFiles/CSGOptiXPTX.dir/CSGOptiX7.ptx.d 
     -x cu 
     -ptx 
      /home/blyth/opticks/CSGOptiX/CSGOptiX7.cu 
     -o CMakeFiles/CSGOptiXPTX.dir/CSGOptiX7.ptx



from nvcc -h > /tmp/nvcc.help


-g
    Generate debug information for host code

-MD
    switch on dependency file output

-MF CMakeFiles/CSGOptiXPTX.dir/CSGOptiX7.ptx.d 
    specify path of dependency file

-MT CMakeFiles/CSGOptiXPTX.dir/CSGOptiX7.ptx 
    specify the target name in the generated dependency file\

-x cu
   specify language for input files

-ptx
   compile .cu to .ptx     






EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

cu=${BASH_SOURCE/.sh/.cu}
touch $cu
VERBOSE=1 om 



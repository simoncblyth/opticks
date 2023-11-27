#!/bin/bash -l 
usage(){ cat << EOU
storch_test.sh
================

CPU test of CUDA code to generate torch photons using s_mock_curand.h::

   ~/opticks/sysrap/tests/storch_test.sh build
   ~/opticks/sysrap/tests/storch_test.sh run
   ~/opticks/sysrap/tests/storch_test.sh ana

   ~/opticks/sysrap/tests/storch_test.sh build_run_ana   


HMM : unless can get the implementation to work header only its 
kinda pointless not including storch_test in the CMake built 
ones as normally need to rebuild sysrap anyhow. 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

cd $(dirname $BASH_SOURCE)

case $(uname) in 
   Darwin) defarg=build_env_run_ana ;; 
   Linux)  defarg=build_env_run ;; 
esac
arg=${1:-$defarg}


TMP=${TMP:-/tmp/$USER/opticks}


msg="=== $BASH_SOURCE :"
name=storch_test 
bdir=$TMP/$name.build
mkdir -p $bdir
bin=$bdir/$name

script=$name.py 

vers=sphere
#vers=sphere_marsaglia
#vers=sphere__restricted
#vers=sphere_marsaglia__restricted


#vers=circle_outwards
#vers=circle_inwards
#vers=M1up99

# HMM: generating into InputPhotons somewhat dirty 
# odir=$HOME/.opticks/InputPhotons/storch
# fold=$odir/$vers
# mkdir -p $fold

fold=$TMP/$name
export FOLD=$fold   # controls where generated photons will be saved
mkdir -p $FOLD


vars="BASH_SOURCE name bdir bin script FOLD vers"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi 

if [ "$vers" == "up" ]; then 

    export storch_FillGenstep_pos=0,0,-990
    export storch_FillGenstep_mom=0,0,1
    export storch_FillGenstep_radius=49

elif [ "$vers" == "up99" -o "$vers" == "M1up99" ]; then 

    export storch_FillGenstep_pos=0,0,-99
    export storch_FillGenstep_mom=0,0,1
    export storch_FillGenstep_radius=49

elif [ "$vers" == "down" ]; then 

    export storch_FillGenstep_pos=0,0,990
    export storch_FillGenstep_mom=0,0,-1
    export storch_FillGenstep_radius=49

elif [ "$vers" == "circle_outwards" ]; then 

    export storch_FillGenstep_type=circle
    export storch_FillGenstep_radius=1
    export storch_FillGenstep_pos=0,0,0

elif [ "$vers" == "circle_inwards" ]; then 

    export storch_FillGenstep_type=circle
    export storch_FillGenstep_radius=-100
    export storch_FillGenstep_pos=0,0,0

elif [ "$vers" == "sphere" -o "$vers" == "sphere_marsaglia" ]; then 

    export storch_FillGenstep_type=$vers
    export storch_FillGenstep_radius=100    # -ve radius for inwards
    export storch_FillGenstep_pos=0,0,0
    export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)

elif [    "$vers" == "sphere__restricted" 
       -o "$vers" == "sphere_marsaglia__restricted"
     ]; then 

    export storch_FillGenstep_type=${vers/__*} # string before __
    export storch_FillGenstep_radius=100   # -ve radius for inwards
    export storch_FillGenstep_pos=0,0,0
    export storch_FillGenstep_zenith=0.25,0.75
    export storch_FillGenstep_azimuth=0.25,0.75
    export storch_FillGenstep_distance=0.25 # frac_twopi control of polarization direction

fi 



K1=1000
K10=10000 
K100=100000
M1=1000000

case ${vers:-dummy} in 
  circle*) num=100 ;; 
  K100*) num=$K100 ;;   ## NB must start with most specific prefix 
  K10*)  num=$K10  ;; 
  K1*)   num=$K1   ;; 
  M1*)   num=$M1   ;; 
    *)   num=$K10  ;; 
esac
export SEvent_MakeGenstep_num_ph=$num
export SEvent_MakeGenstep_dump=1 


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/build}" != "$arg" ]; then 

    case $(uname) in 
        Darwin) libline="-L$OPTICKS_PREFIX/lib" ;;
        Linux) libline="-L$OPTICKS_PREFIX/lib64 -lm -lssl -lcrypto" ;; 
    esac

    gcc $name.cc \
        -o $bin \
        -std=c++11 -lstdc++ \
        -DMOCK_CURAND \
        -I.. \
        -I$CUDA_PREFIX/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/plog/include \
        $libline \
        -lSysRap 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 


if [ "${arg/env}" != "$arg" ]; then 
   env | grep storch_FillGenstep
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi

if [ "${arg/grab}" != "$arg" ]; then 
    echo $msg odir $odir

    xdir=$odir/       ## require trailing slash to avoid rsync duplicating path element 
    from=P:$xdir
    to=$xdir

    vars="xdir from to"
    dumpvars(){ for var in $vars ; do printf "%-30s : %s \n" $var "${!var}" ; done ; } 
    dumpvars
    read -p "$msg Enter YES to proceed with rsync between from and to " ans 
    if [ "$ans" == "YES" ]; then 
        echo $msg proceeding 
        mkdir -p $to 
        rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
        tto=${to%/}  # trim the trailing slash 
        find $tto -name '*.json' -o -name '*.txt' -print0 | xargs -0 ls -1rt 
        echo ls.1 tto $tto jpg mp4 npy 
        find $tto -name '*.jpg' -o -name '*.mp4' -o -name '*.npy' -print0 | xargs -0 ls -1rt
    else
       echo $msg skipping
    fi  
fi

if [ "${arg/ls}" != "$arg" ]; then
   tto=$odir
   find $tto -name '*.json' -o -name '*.txt' -print0 | xargs -0 ls -1rt 
   find $tto -name '*.jpg' -o -name '*.mp4' -o -name '*.npy' -print0 | xargs -0 ls -1rt
fi 


exit 0 



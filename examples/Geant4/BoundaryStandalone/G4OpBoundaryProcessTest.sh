#!/bin/bash -l 

usage(){ cat << EOU
G4OpBoundaryProcessTest.sh
============================

Usage::

   ./G4OpBoundaryProcessTest.sh 
       ## build and run the Geant4 standalone propagate_at_boundary test 
       ## S or P polarized switch is in the script

   ./G4OpBoundaryProcessTest.sh cf 
       ## simple numpy a-b comparison of random aligned Opticks and Geant4 propagate_at_boundary 


Standalone testing of bouundary process using a modified G4OpBoundaryProcess_MOCK
with externally set surface normal. 

NB this somewhat naughtily uses the NP.hh direct from $HOME/np not the one from sysrap, 
the reason is that these standalone tests generally avoid using much Opticks code 
in an effort to stay quite standalone and usable without much installation effort.
Because of that there is more dependence on NP so it tends to be a good environment 
to add functionality to NP, making direct use easier for development. 

Get $HOME/np/NP.hh by cloning  : https://github.com/simoncblyth/np/

EOU
}

msg="=== $BASH_SOURCE :"

srcs=(
    G4OpBoundaryProcessTest.cc 
    G4OpBoundaryProcess_MOCK.cc
     ../CerenkovStandalone/OpticksUtil.cc
     ../CerenkovStandalone/OpticksRandom.cc
   )

name=${srcs[0]}
name=${name/.cc}

for src in $srcs ; do echo $msg $src ; done

g4-
clhep-
boost-
cuda-   # just for vector_functions.h 

standalone-compile(){ 
    local name=$1
    name=${name/.cc}
    mkdir -p /tmp/$name

    cat << EOC
    gcc \
        $* \
         -std=c++11 \
       -I. \
       -I../CerenkovStandalone \
       -I../../../extg4 \
       -I../../../sysrap \
       -DMOCK \
       -DMOCK_DUMP \
       -g \
       -I$HOME/np \
       -I$(cuda-prefix)/include  \
       -I$(boost-prefix)/include \
       -I$(g4-prefix)/include/Geant4 \
       -I$(clhep-prefix)/include \
       -L$(g4-prefix)/lib \
       -L$(g4-prefix)/lib64 \
       -L$(clhep-prefix)/lib \
       -L$(boost-prefix)/lib \
       -lstdc++ \
       -lboost_system \
       -lboost_filesystem \
       -lG4global \
       -lG4materials \
       -lG4particles \
       -lG4track \
       -lG4tracking \
       -lG4processes \
       -lG4geometry \
       -lCLHEP \
       -o /tmp/$name/$name 
EOC
}


arg=${1:-build_run_ana}



seqpath="/tmp/$USER/opticks/QSimTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
#seqpath=$seqpath/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy     ## first tenth of full 256M randoms 
# comment last list to concatenate all 10 tranches giving full 256M randoms allowing num_photons max of 1M
export OPTICKS_RANDOM_SEQPATH=$seqpath

M1=1000000
K3=100000   # 100k is limit when using a single file OPTICKS_RANDOM_SEQPATH
num=$M1  
#num=8
nrm=0,0,1

test=propagate_at_boundary
#test=propagate_at_boundary_normal_incidence

#test=propagate_at_boundary_s_polarized
#test=propagate_at_boundary_p_polarized
#test=propagate_at_boundary_x_polarized

export TEST=${TEST:-$test}
export NUM=${NUM:-$num}
export NRM=${NRM:-$nrm}
#DEBUG=1

case $TEST in 
    propagate_at_boundary)                  src=ephoton                  ;;
    propagate_at_boundary_normal_incidence) src=ephoton             ;;
    propagate_at_boundary_s_polarized) src=hemisphere_s_polarized   ;;
    propagate_at_boundary_p_polarized) src=hemisphere_p_polarized   ;;
    propagate_at_boundary_x_polarized) src=hemisphere_x_polarized   ;;
esac

case $TEST in 
   propagate_at_boundary*) script_stem=propagate_at_boundary ;;
esac

script_dir=../../../qudarap/tests
script=$script_dir/${script_stem}.py
script_cf=$script_dir/${script_stem}_cf.py


qutdir=../../../qudarap/tests
source $qutdir/fill_state.sh 

if [ "$src" == "ephoton" ]; then
    srcdir=
    source $qutdir/ephoton.sh 
else
    srcdir=/tmp/$USER/opticks/QSimTest/$src
    export OPTICKS_BST_SRCDIR=$srcdir
    [ ! -d "$OPTICKS_BST_SRCDIR" ] && echo $msg ERROR OPTICKS_BST_SRCDIR $OPTICKS_BST_SRCDIR does not exist && exit 1 
fi 

q_dstdir=/tmp/$USER/opticks/QSimTest/$TEST
dstdir=/tmp/$USER/opticks/G4OpBoundaryProcessTest/$TEST

export OPTICKS_QSIM_DSTDIR=$q_dstdir
export OPTICKS_BST_DSTDIR=$dstdir

export FOLD=$OPTICKS_BST_DSTDIR
export A_FOLD=$OPTICKS_QSIM_DSTDIR
export B_FOLD=$OPTICKS_BST_DSTDIR
mkdir -p $OPTICKS_BST_DSTDIR


if [ "${arg/cf}" != "$arg" ]; then 

   [ ! -f "$A_FOLD/p.npy" ] && echo $msg ERROR A_FOLD $A_FOLD does not contain p.npy  && exit 1 
   [ ! -f "$B_FOLD/p.npy" ] && echo $msg ERROR B_FOLD $B_FOLD does not contain p.npy  && exit 1 

   echo $msg script_cf $script_cf A_FOLD $A_FOLD B_FOLD $B_FOLD
   ${IPYTHON:-ipython} --pdb -i $script_cf
   [ $? -ne 0 ] && echo $msg cf error && exit 2 
   echo $msg script_cf $script_cf
   exit 0 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    standalone-compile ${srcs[@]}
    eval $(standalone-compile ${srcs[@]})
    [ $? -ne 0 ] && echo $msg compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 

    if [ -n "$DEBUG" ]; then 
        BP=G4OpBoundaryProcess_MOCK::PostStepDoIt lldb__ /tmp/$name/$name
    else
        /tmp/$name/$name
    fi 
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    echo $msg ana script $script FOLD $FOLD
    ${IPYTHON:-ipython} --pdb -i $script   
    [ $? -ne 0 ] && echo $msg ana error && exit 3
    echo $msg ana script $script FOLD $FOLD

fi 

exit 0 




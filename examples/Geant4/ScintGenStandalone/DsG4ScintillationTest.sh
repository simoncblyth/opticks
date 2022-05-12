#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

srcs=(
    DsG4ScintillationTest.cc 
    DsG4Scintillation.cc 
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

    local opt="-DSTANDALONE "

    cat << EOC
    gcc \
        $* \
         -std=c++11 \
       -I. \
       -I../CerenkovStandalone \
       -I../../../sysrap \
       -I../../../u4 \
       -I../../../extg4 \
       -I../../../qudarap \
       $opt \
       -g \
       -I$(cuda-prefix)/include  \
       -I$(boost-prefix)/include \
       -I$(opticks-prefix)/externals/plog/include \
       -I$(opticks-prefix)/externals/glm/glm \
       -I$(g4-prefix)/include/Geant4 \
       -I$(clhep-prefix)/include \
       -L$(g4-prefix)/lib \
       -L$(clhep-prefix)/lib \
       -L$(boost-prefix)/lib \
       -L$(opticks-prefix)/lib \
       -lstdc++ \
       -lboost_system \
       -lboost_filesystem \
       -lSysRap \
       -lU4 \
       -lG4global \
       -lG4materials \
       -lG4particles \
       -lG4track \
       -lG4tracking \
       -lG4processes \
       -lG4geometry \
       -lG4intercoms \
       -lCLHEP \
       -o /tmp/$name/$name 
EOC
}

arg=${1:-build_run_ana}


if [ "${arg/build}" != "$arg" ]; then
    standalone-compile ${srcs[@]}
    eval $(standalone-compile ${srcs[@]})
    [ $? -ne 0 ] && echo $msg compile error && exit 1
fi

export DsG4Scintillation_verboseLevel=2 


if [ "${arg/dbg}" != "$arg" ]; then
    BP=DsG4Scintillation::PostStepDoIt lldb__ /tmp/$name/$name
    [ $? -ne 0 ] && echo $msg dbg error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then
    script=$name.py 
    echo $msg ana script $script FOLD $FOLD

    if [ -f "$script" ]; then
        ${IPYTHON:-ipython} --pdb -i $script
        [ $? -ne 0 ] && echo $msg ana error && exit 3
        echo $msg ana script $script FOLD $FOLD
    else
        echo $msg ERROR script $script does not exist && exit 4
    fi
fi

exit 0 





#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

srcs=(G4OpBoundaryProcessTest.cc ../CerenkovStandalone/OpticksUtil.cc)
name=${srcs[0]}
name=${name/.cc}

g4-
clhep-
boost-

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
       -g \
       -I$HOME/np \
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

DEBUG=1

if [ "${arg/build}" != "$arg" ]; then 
    standalone-compile ${srcs[@]}
    eval $(standalone-compile ${srcs[@]})
    [ $? -ne 0 ] && echo $msg compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 

    if [ -n "$DEBUG" ]; then 
        BP=G4OpBoundaryProcess::PostStepDoIt lldb__ /tmp/$name/$name
    else
        /tmp/$name/$name
    fi 
    [ $? -ne 0 ] && echo $msg run error && exit 2 

fi 





if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py   
    [ $? -ne 0 ] && echo $msg ana error && exit 3
fi 

exit 0 




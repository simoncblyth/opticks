#!/bin/bash -l 

g4-
clhep-
boost-

# clhep- overrides  as clhep-prefix/ver do not match what the geant4 was built against 
clhep-prefix(){  echo /usr/local/opticks_externals/clhep ; }
clhep-ver(){    echo 2.4.1.0 ; }


usage(){ cat << EOU

In addition to geant4 and clhep this also uses 
the NP.hh header from https://github.com/simoncblyth/np/ 

EOU
}

compile(){ 
    local subj=$1 ; 
    local name=$2 ; 
    local extr=$3 ; 
    mkdir -p /tmp/$name
    cat << EOC
    gcc \
        $subj.cc \
        $name.cc \
        $extr.cc \
        -DINSTRUMENTED \
        -DSKIP_CONTINUE \
         -std=c++11 \
       -I. \
       -g \
       -I$HOME/np \
       -I$(boost-prefix)/include \
       -I$(g4-prefix)/include/Geant4 \
       -I$(clhep-prefix)/include \
       -L$(g4-prefix)/lib \
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
       -lCLHEP-$(clhep-ver) \
       -o /tmp/$name/$name 
EOC
}


run(){  
    local name=$1 ; 
    local var 
    case $(uname) in 
       Darwin) var=DYLD_LIBRARY_PATH ;;
        Linux) var=LD_LIBRARY_PATH ;;
    esac

    case $(uname) in 
       Darwin) DEBUG=lldb__ ;;
        Linux) DEBUG=gdb    ;;
    esac
    #DEBUG=
    #export BP=G4Cerenkov_modified::PostStepDoIt

    cat << EOC
$var=$(boost-prefix)/lib:$(g4-prefix)/lib:$(clhep-prefix)/lib $DEBUG /tmp/$name/$name 
EOC

}



subj=G4Cerenkov_modified
name=${subj}Test
extr=OpticksDebug

if [ -n "$SCAN" ]; then 
    docompile=0
    interactive=0
else
    docompile=1
    interactive=1
fi 


if [ $docompile -eq 1 ]; then
    compile $subj $name $extr
    eval $(compile $subj $name $extr)
    [ $? -ne 0 ] && echo compile FAIL && exit 1 
fi 


run $name 
eval $(run $name) $*
[ $? -ne 0 ] && echo run FAIL && exit 2
echo run succeeds




if [ $interactive -eq 1 ]; then 
    ipython -i $name.py
    [ $? -ne 0 ] && echo analysis FAIL && exit 3
else
    python $name.py
    [ $? -ne 0 ] && echo analysis FAIL && exit 3
fi



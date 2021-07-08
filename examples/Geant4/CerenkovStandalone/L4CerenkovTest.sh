#!/bin/bash  -l

g4-
clhep-

# clhep- overrides  as clhep-prefix/ver do not match what the geant4 was built against 
clhep-prefix(){  echo /usr/local/opticks_externals/clhep ; }
clhep-ver(){    echo 2.4.1.0 ; }


usage(){ cat << EOU

In addition to geant4 and clhep this also uses 
the NP.hh header from https://github.com/simoncblyth/np/ 

EOU
}

compile(){ 
    local name=$1 ; 
    mkdir -p /tmp/$name
    cat << EOC
    gcc $name.cc -std=c++11 \
       -I. \
       -I$HOME/np \
       -I$(g4-prefix)/include/Geant4 \
       -I$(clhep-prefix)/include \
       -L$(g4-prefix)/lib \
       -L$(clhep-prefix)/lib \
       -lstdc++ \
       -lG4global \
       -lG4materials \
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

    cat << EOC
$var=$(g4-prefix)/lib:$(clhep-prefix)/lib /tmp/$name/$name 
EOC
}


name=L4CerenkovTest

compile $name
eval $(compile $name)
[ $? -ne 0 ] && echo compile FAIL && exit 1 

run $name
eval $(run $name)
[ $? -ne 0 ] && echo run FAIL && exit 2

echo run succeeds

exit 0 


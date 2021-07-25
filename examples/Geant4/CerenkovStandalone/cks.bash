cks-vi(){ vi $BASH_SOURCE ; }
cks-env(){
    g4-
    clhep-
    boost-
}

cks-usage(){ cat << EOU

In addition to geant4 and clhep this also uses 
the NP.hh header from https://github.com/simoncblyth/np/ 

::


EOU
}

cks-compile(){ 
    local name=$1
    name=${name/.cc}
    mkdir -p /tmp/$name

    cat << EOC
    gcc \
        $* \
        -DINSTRUMENTED \
        -DFLOAT_TEST \
        -DSKIP_CONTINUE \
         -std=c++11 \
       -I. \
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
       -lCLHEP \
       -o /tmp/$name/$name 
EOC
}


cks-run(){  
    local name=$1 ; 
    name=${name/.cc}
    mkdir -p /tmp/$name

    local var 
    case $(uname) in 
       Darwin) var=DYLD_LIBRARY_PATH ;;
        Linux) var=LD_LIBRARY_PATH ;;
    esac

    case $(uname) in 
       Darwin) DEBUG=lldb__ ;;
        Linux) DEBUG=gdb    ;;
    esac
    unset DEBUG

    mkdir -p /tmp/$name

    cat << EOC
$var=$(boost-prefix)/lib:$(g4-prefix)/lib:$(g4-prefix)/lib64:$(clhep-prefix)/lib $DEBUG /tmp/$name/$name 
EOC

}



#!/bin/bash -l 

usage(){ cat << EOU
examples/UsePlogStandalone/go.sh
==================================

This attempts to reproduce some plog misbehaviour without success

Note however that the latest plog has a PLOG_LOCAL switch that 
may avoid the need for visibility hidden and clashes
when not using that. 

HMM: the JUNOSW issue might be because there is no access to main 
everything is in shared libs .. the main is python.  
So that perhaps gives problem of static initialization ordering ?

EOU
}


msg="=== $BASH_SOURCE :"
name=UsePlogStandalone
srcdir=$PWD
tmpdir=/tmp/$USER/opticks/$name
mkdir -p $tmpdir
mkdir -p $tmpdir/obj

defarg="plog_build_run"
arg=${1:-$defarg}

if [ "${arg/plog}" != "$arg" ]; then


    cd $tmpdir
    pwd

    url=https://github.com/simoncblyth/plog.git
    #url=https://github.com/SergiusTheBest/plog

    URL=${URL:-$url}
    echo $msg URL $URL 

    if [ -d "plog" ]; then
        echo $msg plog has already been cloned into $tmpdir/plog 
        echo $msg from URL : $(grep url $tmpdir/plog/.git/config) 
        echo $msg delete that plog folder to clone again from different URL
    else
        echo $msg cloning from URL $URL   
        git clone $URL
    fi 
fi 


compile(){ cat << EOL
gcc -g -c -Wall -Werror -fpic $1.cc -o $tmpdir/obj/$1.o -std=c++11  -fvisibility=hidden -fvisibility-inlines-hidden -I$tmpdir/plog/include 
EOL
}

makelib(){ cat << EOL
gcc -shared -o $tmpdir/libDEMO.dylib -lstdc++ $tmpdir/obj/DEMO.o $tmpdir/obj/DEMO_LOG.o 
EOL
}

makemain(){ cat << EOL
gcc -g -Wall -Werror UsePlogStandalone.cc -o $tmpdir/UsePlogStandalone \
     -I$tmpdir/plog/include \
    -std=c++11 -fvisibility=hidden -fvisibility-inlines-hidden  \
    -lstdc++ -L$tmpdir -lDEMO 
EOL
}

if [ "${arg/build}" != "$arg" ]; then

    cd $srcdir
    pwd

    echo $msg compile with c++11

    srcs="DEMO DEMO_LOG"
    for src in $srcs 
    do 
        cmd=$(compile $src) 
        echo $cmd
        eval $cmd
        [ $? -ne 0 ] && echo $msg $src compilation error && exit 1 
    done 

    cmd=$(makelib)
    echo $cmd
    eval $cmd 
    [ $? -ne 0 ] && echo $msg make lib  error && exit 1 

    cmd=$(makemain)
    echo $cmd
    eval $cmd 
    [ $? -ne 0 ] && echo $msg make main  error && exit 1 



fi 


if [ "${arg/run}" != "$arg" ]; then
    export PLOGPATH=$tmpdir/$name.log 
    rm $PLOGPATH

    $tmpdir/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2
    echo $msg cat $PLOGPATH
    cat $PLOGPATH
fi


exit 0 

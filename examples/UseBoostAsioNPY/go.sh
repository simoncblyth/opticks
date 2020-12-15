#!/bin/bash -l

TMPDIR=/tmp/$USER/UseBoostAsioNPY

chat-build()
{
   local msg="=== $FUNCNAME :"
   boost-
   local exe=$TMPDIR/${1/.cpp}
   mkdir -p $(dirname $exe)
   echo $msg compiling $1 to yield $exe
   g++ -o $exe \
    -I. \
    -Wno-deprecated-declarations \
    -I$(boost-prefix)/include \
     $1 \
    -L$(boost-prefix)/lib \
    -lboost_system -lboost_thread 
}
chat-usage(){ cat << EOU

In one session start server::

   $TMPDIR/chat_server 8080

In another session start client and type "hello"::

   $TMPDIR/chat_client 127.0.0.1 8080

EOU
}


build-om()
{
    local sdir=$(pwd)
    local bdir=/tmp/$USER/opticks/$(basename $sdir)/build 
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

    om-
    om-cmake $sdir  
    make
    make install   
}


#chat-build chat_server.cpp 
#chat-build chat_client.cpp 
#chat-usage

build-om



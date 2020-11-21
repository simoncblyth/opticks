#!/bin/bash -l

exe=/tmp/$USER/ListenUDPTest

build-manual()
{
   boost-
   local exe=$1
   mkdir -p $(dirname $exe)
   g++ -o $exe \
    -I. \
    -I$(boost-prefix)/include \
     ListenUDP.cc \
     MockViz.cc \
     tests/ListenUDPTest.cc \
    -L$(boost-prefix)/lib \
    -lboost_system -lboost_thread -lpthread
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


build-om && which ListenUDPTest && ListenUDPTest
#build-manual $exe && $exe


usage(){ cat << EOU
Send udp messages to the ListenUDPTest server with eg::

   UDP_PORT=15001 udp.py hello from udp.py 

Using ~/env/bin/udp.py 

EOU
}


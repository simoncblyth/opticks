#!/bin/bash -l

boost-

exe=/tmp/$USER/ListenUDPTest
mkdir -p $(dirname $exe)

g++ -o $exe \
    -I. \
    -I$(boost-prefix)/include \
     ListenUDP.cc \
     ListenUDPTest.cc \
     Viz.cc \
    -L$(boost-prefix)/lib \
    -lboost_system -lboost_thread && $exe


usage(){ cat << EOU

Send udp messages to this with eg::

   UDP_PORT=15001 udp.py hello from udp.py 



EOU
}


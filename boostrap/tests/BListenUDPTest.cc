/**
BListenUDPTest.cc
==================

This demonstrates how to integrate async-io for handling 
UDP messages with a fake GUI vizualization loop. 
Using io_context::poll_one

For info on Boost ASIO see env-;basio- 

This test started from examples/UseBoostAsioUDP/ListenUDPTest.cc

Send UDP messages to this with eg ~/env/bin/udp.py::

   UDP_PORT=15001 udp.py hello from udp.py 


Have observed that PLOG logging does not provide any output 
from within asio callback handlers. Simple std::cout does
work from within handlers.

**/

#include <iostream>
#include <boost/asio.hpp>

#include "SSys.hh"
#include "SMockViz.hh"
#include "BListenUDP.hh"

template class BListenUDP<SMockViz>;

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    if(SSys::IsCTestRunning())
    {
        LOG(info) << "This test is not appropriate for CTest running as its a UDP server " ; 
        return 0 ; 
    }

    SMockViz viz ; 

    boost::asio::io_context io ; 
    BListenUDP<SMockViz> listen(io, &viz) ; 

    unsigned count = 0 ; 
    unsigned maxcount = 0x1 << 31 ; 

    while(count < maxcount) {
       if( count % 1000000 == 0) std::cout << count << std::endl ; 
       count++ ; 
       io.poll_one();   // handlers observed to run on main thread 
    }

    return 0;
}


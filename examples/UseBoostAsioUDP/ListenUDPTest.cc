/**
ListenUDPTest.cc
=================

This demonstrates how to integrate async-io for handling 
UDP messages with a fake GUI vizualization loop. 
Using io_context::poll_one

For info on Boost ASIO see env-;basio- 

**/

#include <iostream>
#include <boost/asio.hpp>

#include "Viz.hh"
#include "ListenUDP.hh"


int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    Viz vz ; 

    boost::asio::io_context io ; 
    ListenUDP listen(io, &vz) ; 

    unsigned count = 0 ; 
    while(1) {
       //if( count % 100000 == 0) std::cout << count << std::endl ; 
       count++ ; 
       io.poll_one();   // handlers observed to run on main thread 
    }

    return 0;
}


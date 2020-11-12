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

#include "MockViz.hh"
#include "ListenUDP.hh"

template class ListenUDP<MockViz>;


int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    MockViz viz ; 

    boost::asio::io_context io ; 
    ListenUDP<MockViz> listen(io, &viz) ; 

    unsigned count = 0 ; 
    while(1) {
       //if( count % 100000 == 0) std::cout << count << std::endl ; 
       count++ ; 
       io.poll_one();   // handlers observed to run on main thread 
    }

    return 0;
}


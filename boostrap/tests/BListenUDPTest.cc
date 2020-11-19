/**
BListenUDPTest.cc
==================

This demonstrates how to integrate async-io for handling 
UDP messages with a fake GUI vizualization loop, using 
io_context::poll_one which does not block thanks to boost::asio
doing the blocking io for you behind the scenes.

UDP messages received as passed to the delegates SCtrl::command method.

Run the UDP listener with::

   BListenUDPTest

From another terminal (actually can be from a different node) send
UDP messages to the listener with eg ~/env/bin/udp.py::

   UDP_PORT=15001 udp.py hello from udp.py $(hostname) $(date) 

Note that everything is happening on the main thread so there is no 
danger of contention or any concerns about locking.

* for info on Boost ASIO see env-;basio- 
* this test started from examples/UseBoostAsioUDP/ListenUDPTest.cc

Initial problems with PLOG logging from within boost::asio handlers 
turned out to be pilot error.  

**/

#include <iostream>

#include "PLOG.hh"
#include "SSys.hh"
#include "SMockViz.hh"
#include "OPTICKS_LOG.hh"

#ifdef WITH_BOOST_ASIO
#include <boost/asio.hpp>
#include "BListenUDP.hh"
template class BListenUDP<SMockViz>;
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << argv[0] ; 

    if(SSys::IsCTestRunning())
    {
        LOG(info) << "This test is not appropriate for CTest running as its a UDP server " ; 
        return 0 ; 
    }

    SMockViz viz ; 


#ifdef WITH_BOOST_ASIO
    boost::asio::io_context io ; 
    BListenUDP<SMockViz> listen(io, &viz) ; 
#else
    LOG(error) << "This test does nothing useful without boost::asio : try: boost-;boost-rebuild-with-asio " ; 
    return 0 ; 
#endif

    unsigned count = 0 ; 
    unsigned maxcount = 0x1 << 31 ; 

    while(count < maxcount) {
       if( count % 1000000 == 0) std::cout << count << std::endl ; 
       count++ ; 
#ifdef WITH_BOOST_ASIO
       io.poll_one();   // handlers observed to run on main thread 
#endif
    }

    return 0;
}


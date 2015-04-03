/*
Test with eg::

        udpserver-test 
        UDP_PORT=13 udp.py hello


        npysend.sh --tag 1  # requires zmq- ; zmq-broker 

*/

#include "numpydelegate.hpp"
#include "numpyserver.hpp"

int main()
{

    numpydelegate nde ;  // example numpydelegate

    numpyserver<numpydelegate> srv(&nde, 8080, "tcp://127.0.0.1:5002");
    
    // 
    // Instanciation of numpyserver spins off a background network thread 
    // that listens for UDP and ZMQ connections.
    // The template type and first argument identify the type and instance
    // of the delegate to me messaged.
    // Other arguments identify the network locations.
    //
    // When messages arrive the numpydelegate *on_msg* or *on_npy* 
    // handlers are invoked back on the main thread during
    // calls to one of the pollers:: 
    //
    //       srv.poll_one()  
    //       srv.poll()  
    //
    // These should be incorporated into the runloop, 
    // they do not block.
    // 

    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        srv.poll();
        //srv.poll_one();
        srv.sleep(1);
    }
    srv.stop();

    return 0;
}


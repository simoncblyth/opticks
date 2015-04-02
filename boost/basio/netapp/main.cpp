/*
Test with eg::

        udpserver-test 
        UDP_PORT=13 udp.py hello


        npysend.sh --tag 1  # requires zmq- ; zmq-broker 

*/

#include "app.hpp"
#include "net_server.hpp"

int main()
{

    App app ;


    net_server<App> srv(&app, 13, "tcp://127.0.0.1:5002");

    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        srv.poll_one();
        srv.sleep(1);
    }
    srv.stop();

    return 0;
}


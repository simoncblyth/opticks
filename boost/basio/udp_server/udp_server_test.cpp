//  clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt udp_server_test.cpp udp_server.cpp -o udp_server_test

/*
Test with python client::

    UDP_PORT=13 udp.py hello
*/


#include "udp_server.hpp"

int main()
{
    unsigned int port = 13 ; 
    try
    {
        boost::asio::io_service io_service;
        udp_server server(io_service, port);
        printf("udp_server_test starting io_service.run() \n");
        io_service.run();
        printf("udp_server_test completed io_service.run() \n");
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cassert>

// seems htonl dont need this header
//#include <arpa/inet.h>

/**
hexlify
-------

Dump the bytes of an object 

**/

std::string hexlify(const void* obj, size_t size, bool reverse)
{ 
    const unsigned char * const bytes = static_cast<const unsigned char *>(obj);
    std::stringstream ss ; 
    for(size_t i=0 ; i < size ; i++) ss << std::setw(2) << std::hex << std::setfill('0') << unsigned(bytes[reverse ? size - 1 - i : i]) ; 
    return ss.str(); 
}

/**
test_network_order_ntohl_htonl
--------------------------------

**/

void test_network_order_ntohl_htonl()
{
    uint32_t hostlong = 0x12abcdef ; 
    uint32_t netlong = htonl(hostlong) ; 

    std::cout 
       << " std::hex "
       << " hostlong " << std::hex << hostlong 
       << " netlong " << std::hex << netlong 
       << std::endl 
       ;

    std::cout 
       << " hexlify "
       << " hostlong " << hexlify(&hostlong,4,true)
       << " netlong "  << hexlify(&netlong, 4,true) 
       << std::endl 
       ;

    uint32_t hostlong2 = ntohl(netlong) ;
    uint32_t netlong2 = htonl(hostlong2) ; 

    assert( hostlong == hostlong2 ); 
    assert( netlong == netlong2 ); 
}

int main(int argc, char** argv)
{
    test_network_order_ntohl_htonl();
    return 0 ; 
}

// gcc test_network_order_ntohl_htonl.cc -lstdc++ -o /tmp/test_network_order_ntohl_htonl && /tmp/test_network_order_ntohl_htonl 

#pragma once
#include <cstdint>
#include <string>

#define API  __attribute__ ((visibility ("default")))

/**
npy_header
------------

ABANDONED : Decided ~/np/net_hdr.hh approach is simpler and better that this 

**/

struct API npy_header
{
    static std::string hexlify(const void* obj, size_t size, bool reverse);
    static npy_header unpack( void* data );

    npy_header(unsigned arr_bytes, unsigned meta_bytes );

    uint32_t     arr_bytes() const ;
    uint32_t     meta_bytes() const ;
    char*        data() const ;  
    std::string  desc() const ;

    uint32_t m_arr_bytes ; 
    uint32_t m_meta_bytes ; 
};


#include <sstream>
#include <cassert>
#include <iomanip>
#include <iostream>


std::string npy_header::hexlify(const void* obj, size_t size, bool reverse)  // static 
{ 
    const unsigned char * const bytes = static_cast<const unsigned char *>(obj);
    std::stringstream ss ; 
    for(size_t i=0 ; i < size ; i++) ss << std::setw(2) << std::hex << std::setfill('0') << unsigned(bytes[reverse ? size - 1 - i : i]) ; 
    return ss.str(); 
}

npy_header npy_header::unpack( void* data )  // static
{
    npy_header hdr(0,0); 
    memcpy( &hdr, data, sizeof(hdr) ); 
    return hdr ; 
}
npy_header::npy_header(unsigned arr_bytes, unsigned meta_bytes)
    :
    // stored big endian (network order)
    m_arr_bytes(htonl(arr_bytes)),
    m_meta_bytes(htonl(meta_bytes))
{
} 
uint32_t npy_header::arr_bytes() const 
{
    return ntohl(m_arr_bytes) ; 
}
uint32_t npy_header::meta_bytes() const 
{
    return ntohl(m_meta_bytes) ; 
}
char* npy_header::data() const 
{
    return (char*)this ; 
}

std::string npy_header::desc() const 
{
    char* p = (char*)this ;  
    std::stringstream ss ; 
    ss << "header"
       << " hx0 " << hexlify(p   , 4, true )
       << " hx1 " << hexlify(p+4 , 4, true )
       << " m_arr_bytes " << std::hex << m_arr_bytes
       << " m_meta_bytes " << std::hex << m_meta_bytes
       << " arr_bytes() " << arr_bytes() 
       << " meta_bytes() " << meta_bytes() 
       ;
    return ss.str(); 
}



#pragma once
#include <cstdint>
#include <string>

#define API  __attribute__ ((visibility ("default")))

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




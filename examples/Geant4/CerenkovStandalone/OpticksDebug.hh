#pragma once

#include <vector>
#include <string>
#include <cstdint>


struct UU
{
   uint32_t x ; 
   uint32_t y ; 
};

union DUU
{
   double d ; 
   UU     uu ; 
};



struct HH
{
   uint16_t x ; 
   uint16_t y ; 
};

union FHH
{
   float  f ; 
   HH     hh ; 
};




template <typename T>
struct OpticksDebug
{
    unsigned itemsize ; 
    const char* name ; 

    OpticksDebug(unsigned itemsize, const char* name); 

    std::vector<std::string> names ; 
    std::vector<T>           values ; 

    void append( T x, const char* name );
    void append( unsigned x, unsigned y, const char* name ); 
    void write(const char* dir, const char* reldir, unsigned nj, unsigned nk ) ; 

};



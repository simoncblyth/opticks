#pragma once

/**
SPack
======

Static packing/unpacking utilities.


**/


#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SPack {
     public:
         static unsigned Encode(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
         static unsigned Encode(const unsigned char* ptr, const unsigned num); 

         static void Decode( const unsigned value,  unsigned char& x, unsigned char& y, unsigned char& z, unsigned char& w ); 
         static void Decode( const unsigned value,  unsigned char* ptr, const unsigned num); 
     

};



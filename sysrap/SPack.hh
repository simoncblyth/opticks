#pragma once

/**
SPack
======

Static packing/unpacking utilities.


**/


#include <cstddef>
#include <cstdint>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SPack {
     public:
        union uif_t {
            unsigned u ; 
            int      i ;
            float    f ; 
        }; 

        union ui16_t {
            uint16_t  u ; 
            int16_t   i ; 
        };


     public:
        static bool     IsLittleEndian(); 
        static unsigned Encode(unsigned x, unsigned y, unsigned z, unsigned w);  // values must fit into unsigned char, ie up to  0xff   
        static void     Decode( const unsigned int value,  unsigned& x, unsigned& y, unsigned& z, unsigned& w );

        static unsigned Encode(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
        static unsigned Encode(const unsigned char* ptr, const unsigned num); 

        static void Decode( const unsigned value,  unsigned char& x, unsigned char& y, unsigned char& z, unsigned char& w ); 
        static void Decode( const unsigned value,  unsigned char* ptr, const unsigned num); 

     public:
        static unsigned Encode13(unsigned char c, unsigned int ccc);
        static void Decode13( const unsigned int value, unsigned char& c, unsigned int& ccc );

     public:
        static unsigned Encode22(unsigned a, unsigned b);
        static void     Decode22( const unsigned value, unsigned& a, unsigned& b);
        static unsigned Decode22a( const unsigned value ); 
        static unsigned Decode22b( const unsigned value ); 
     public:
        static unsigned Encode22hilo( int a, int b ); 
        static void     Decode22hilo( const unsigned packed, int& a, int& b ); 
        static int      Decode22hi( const unsigned packed ); 
        static int      Decode22lo( const unsigned packed ); 
     public:
        static float int_as_float( const int i ); 
        static int int_from_float( const float f ); 
        static float uint_as_float( const unsigned f ); 
        static unsigned uint_from_float( const float f ); 

        static float unsigned_as_float( const unsigned u ) ;
        static unsigned float_as_unsigned( const float f ) ;


        template <int NUM_BITS>
        static int unsigned_as_int(unsigned value);

        static int unsigned_as_int_32(unsigned value);
        static int unsigned_as_int_16(unsigned value);

        static double unsigned_as_double( const unsigned x, const unsigned y ) ;
        static void double_as_unsigned(unsigned& x, unsigned& y, const double d ) ;



};



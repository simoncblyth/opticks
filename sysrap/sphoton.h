#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SPHOTON_METHOD __host__ __device__ __forceinline__
#else
#    define SPHOTON_METHOD inline 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
   #include <sstream>
   #include <vector>
   #include <cstring>
   #include <cassert>
#endif



struct sphoton
{
    float3 pos ; 
    float  time ; 

    float3 mom ; 
    float  weight ; 

    float3 pol ; 
    float  wavelength ;   

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif 

    SPHOTON_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
    SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; } 

    SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
    SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 

    SPHOTON_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; }
    SPHOTON_METHOD unsigned boundary() const { return boundary_flag >> 16 ; }

    SPHOTON_METHOD void     set_flag(unsigned flag) {         boundary_flag = ( boundary_flag & 0xffff0000u ) | ( flag & 0xffffu ) ; flagmask |= flag ;  } // clear flag bits then set them  
    SPHOTON_METHOD void     set_boundary(unsigned boundary) { boundary_flag = ( boundary_flag & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ; }        // clear boundary bits then set them 

    SPHOTON_METHOD void zero_flags() { boundary_flag = 0u ; identity = 0u ; orient_idx = 0u ; flagmask = 0u ; } 


    unsigned boundary_flag ; 
    unsigned identity ; 
    unsigned orient_idx ;   
    unsigned flagmask ; 

   /*
    MAYBE RE-ARRANGE PAIRINGS

    unsigned idx ; 
    unsigned orient_identity ; 
    unsigned boundary_flag ;
    unsigned flagmask ; 

    idx always exists for a photon, 
    BUT: orient is only set on intersects together with boundary and identity 
    (plus flag but that is also be set when no intersect)
  
    SO IT WOULD BE BETTER TO HAVE orient_identity and idx alone if identity can spare one bit ?

        identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;

    JUNO max prim_idx ~3245 : so thats OK

    In [1]: 0xffff
    Out[1]: 65535

    In [2]: 0x7fff
    Out[2]: 32767
   */

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline std::string sphoton::desc() const 
{
    std::stringstream ss ; 
    ss 
        << " pos " << pos << " t  " << time << std::endl
        << " mom " << mom << " wg " << weight << std::endl
        << " pol " << pol << " wl " << wavelength << std::endl
        << " bn " << boundary() 
        << " fl " << std::hex << flag() << std::dec
        << " id " << identity 
        << " or " << orient()
        << " ix " << idx() 
        << " fm " << std::hex << flagmask  << std::dec 
        << std::endl 
        ;

    std::string s = ss.str(); 
    return s ; 
} 
#endif 



union qphoton
{
    quad4   q ; 
    sphoton p ; 
}; 




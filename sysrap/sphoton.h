#pragma once
/**
sphoton.h
============

TODO: MAYBE RE-ARRANGE FLAG PAIRINGS::

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
**/



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

    unsigned boundary_flag ; 
    unsigned identity ; 
    unsigned orient_idx ;   
    unsigned flagmask ; 


    SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient );

    SPHOTON_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
    SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; } 

    SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
    SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 

    SPHOTON_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; }
    SPHOTON_METHOD unsigned boundary() const { return boundary_flag >> 16 ; }

    SPHOTON_METHOD void     set_flag(unsigned flag) {         boundary_flag = ( boundary_flag & 0xffff0000u ) | ( flag & 0xffffu ) ; flagmask |= flag ;  } // clear flag bits then set them  
    SPHOTON_METHOD void     set_boundary(unsigned boundary) { boundary_flag = ( boundary_flag & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ; }        // clear boundary bits then set them 

    SPHOTON_METHOD void zero_flags() { boundary_flag = 0u ; identity = 0u ; orient_idx = 0u ; flagmask = 0u ; } 

    SPHOTON_METHOD float* data() {               return &pos.x ; }
    SPHOTON_METHOD const float* cdata() const {  return &pos.x ; }

    SPHOTON_METHOD void zero()
    { 
       pos.x = 0.f ; pos.y = 0.f ; pos.z = 0.f ; time = 0.f ; 
       mom.x = 0.f ; mom.y = 0.f ; mom.z = 0.f ; weight = 0.f ; 
       pol.x = 0.f ; pol.y = 0.f ; pol.z = 0.f ; wavelength = 0.f ; 
       boundary_flag = 0u ; identity = 0u ; orient_idx = 0u ; flagmask = 0u ;  
    }

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    SPHOTON_METHOD std::string desc() const ; 
    SPHOTON_METHOD void ephoton() ; 
    SPHOTON_METHOD void normalize_mom_pol(); 
    SPHOTON_METHOD void transverse_mom_pol(); 
    SPHOTON_METHOD static sphoton make_ephoton(); 
#endif 

}; 

SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_ )
{
    set_boundary(boundary_); 
    identity = identity_ ; 
    set_orient( orient_ );  
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

SPHOTON_METHOD std::string sphoton::desc() const 
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

/**
sphoton::ephoton
---------------

*ephoton* is used from qudarap/tests/QSimTest generate tests as the initial photon, 
which gets persisted to p0.npy 
The script qudarap/tests/ephoton.sh sets the envvars defining the photon 
depending on the TEST envvar. 
 
**/

SPHOTON_METHOD void sphoton::ephoton() 
{
    quad4& q = (quad4&)(*this) ; 

    qvals( q.q0.f ,        "EPHOTON_POST" , "0,0,0,0" );                      // position, time
    qvals( q.q1.f, q.q2.f, "EPHOTON_MOMW_POLW", "1,0,0,1,0,1,0,500" );  // direction, weight,  polarization, wavelength 
    qvals( q.q3.i ,        "EPHOTON_FLAG", "0,0,0,0" );   
    normalize_mom_pol(); 
    transverse_mom_pol(); 
}

SPHOTON_METHOD void sphoton::normalize_mom_pol() 
{
    mom = normalize(mom); 
    pol = normalize(pol); 
}

SPHOTON_METHOD void sphoton::transverse_mom_pol() 
{
    float mom_pol = fabsf( dot(mom, pol)) ;  
    float eps = 1e-5 ; 
    bool is_transverse = mom_pol < eps ; 

    if(!is_transverse )
    {
        std::cout 
             << " sphoton::transverse_mom_pol " 
             << " FAIL "
             << " mom " << mom 
             << " pol " << pol 
             << " mom_pol " << mom_pol 
             << " eps " << eps 
             << " is_transverse " << is_transverse
             << std::endl 
             ; 
    }
    assert(is_transverse); 
}

SPHOTON_METHOD sphoton sphoton::make_ephoton()  // static
{
    sphoton p ; 
    p.ephoton(); 
    return p ; 
}
#endif 


struct sphoton_selector
{
    unsigned hitmask ; 
    sphoton_selector(unsigned hitmask_) : hitmask(hitmask_) {}; 
    SPHOTON_METHOD bool operator() (const sphoton& p) const { return ( p.flagmask & hitmask ) == hitmask  ; }   // require all bits of the mask to be set 
};





union qphoton
{
    quad4   q ; 
    sphoton p ; 
}; 



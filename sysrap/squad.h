#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SQUAD_METHOD __host__ __device__ __forceinline__
#else
#    define SQUAD_METHOD inline 
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


union UIF
{
   float    f ; 
   int      i ; 
   unsigned u ; 
};


union quad
{
   float4 f ; 
   int4   i ; 
   uint4  u ; 
};


inline unsigned int_as_unsigned( int value )
{
   UIF uif ; 
   uif.i = value ; 
   return uif.u ; 
}

inline int unsigned_as_int( unsigned value )
{
   UIF uif ; 
   uif.u = value ; 
   return uif.i ; 
}

struct quad2
{ 
    quad q0 ; 
    quad q1 ;


    SQUAD_METHOD void zero();
    SQUAD_METHOD float* data() ;
    SQUAD_METHOD const float* cdata() const ;

    SQUAD_METHOD float3* normal() ;
    SQUAD_METHOD const float3* normal() const ;
    SQUAD_METHOD float    distance() const ;
    SQUAD_METHOD unsigned identity() const ;
    SQUAD_METHOD unsigned boundary() const ;

    SQUAD_METHOD void set_identity(unsigned id);
    SQUAD_METHOD void set_boundary(unsigned bn);


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static quad2 make_eprd(); 
    void eprd() ; 
#endif
 

}; 

void quad2::zero() 
{
    q0.u.x = 0 ; q0.u.y = 0 ; q0.u.z = 0 ; q0.u.w = 0 ; 
    q1.u.x = 0 ; q1.u.y = 0 ; q1.u.z = 0 ; q1.u.w = 0 ; 
} 

float*         quad2::data() {           return &q0.f.x ;  }
const float*   quad2::cdata() const  {   return &q0.f.x ;  }
float3*        quad2::normal() {         return (float3*)&q0.f.x ;  }
const float3*  quad2::normal() const {   return (float3*)&q0.f.x ;  }
float          quad2::distance() const { return q0.f.w ;  }

unsigned       quad2::identity() const {   return q1.u.z ;  }
void           quad2::set_identity(unsigned id) { q1.u.z = id ;  }
unsigned       quad2::boundary() const {   return q1.u.w ;  }
void           quad2::set_boundary(unsigned bn) { q1.u.w = bn ;  }



struct quad4 
{ 
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ;

    SQUAD_METHOD void zero();
    SQUAD_METHOD float* data() ;
    SQUAD_METHOD const float* cdata() const ;

    // hmm actually *set_flags* doing all at once makes little sense, 
    // as idx does not need to be reset and only determine the flag later 

    SQUAD_METHOD void set_flags(unsigned  boundary, unsigned  identity, unsigned  idx, unsigned  flag, float  orient ); 
    SQUAD_METHOD void set_idx( unsigned  idx); 
    SQUAD_METHOD void set_orient( float orient ); 
    SQUAD_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient ); 
    SQUAD_METHOD void set_flag(unsigned  flag); 

    SQUAD_METHOD void get_flags(unsigned& boundary, unsigned& identity, unsigned& idx, unsigned& flag, float& orient ) const ;
    SQUAD_METHOD void get_idx( unsigned& idx); 
    SQUAD_METHOD void get_orient( float& orient ); 
    SQUAD_METHOD void get_prd( unsigned& boundary, unsigned& identity, float&  orient ); 
    SQUAD_METHOD void get_flag(unsigned& flag) const ; 
    SQUAD_METHOD unsigned flagmask() const ; 

    SQUAD_METHOD void set_wavelength( float wl ); 
    SQUAD_METHOD float wavelength() const ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static quad4 make_ephoton(); 
    void ephoton() ; 
    void normalize_mom_pol(); 
    void transverse_mom_pol(); 
#endif
};


template<typename T>    // T  needs to have flagmask method 
struct qselector
{
    unsigned hitmask ; 
    qselector(unsigned hitmask_) : hitmask(hitmask_) {}; 
    SQUAD_METHOD bool operator() (const T& p) const { return ( p.flagmask() & hitmask ) == hitmask  ; }   // require all bits of the mask to be set 
};


void quad4::zero() 
{
    q0.u.x = 0 ; q0.u.y = 0 ; q0.u.z = 0 ; q0.u.w = 0 ; 
    q1.u.x = 0 ; q1.u.y = 0 ; q1.u.z = 0 ; q1.u.w = 0 ; 
    q2.u.x = 0 ; q2.u.y = 0 ; q2.u.z = 0 ; q2.u.w = 0 ; 
    q3.u.x = 0 ; q3.u.y = 0 ; q3.u.z = 0 ; q3.u.w = 0 ; 
} 

float*       quad4::data() {         return &q0.f.x ;  }
const float* quad4::cdata() const  { return &q0.f.x ;  }



// q3.u.z : orient_idx 

void quad4::set_idx( unsigned  idx)
{
    q3.u.z = ( q3.u.z & 0x80000000u ) | ( 0x7fffffffu & idx ) ;   // retain bit 31 asis 
}
void quad4::get_idx( unsigned& idx)
{
    idx =  q3.u.z & 0x7fffffffu  ;
}
void quad4::set_orient( float orient )  // not typically used as set_prd more convenient, but useful for debug 
{
    q3.u.z =  ( q3.u.z & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; 
}
void quad4::get_orient( float& orient )
{
    orient = ( q3.u.z & 0x80000000u ) ? -1.f : 1.f ;   
}

void quad4::set_prd( unsigned  boundary, unsigned  identity, float  orient )
{
    q3.u.x = ( q3.u.x & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ;  // clear boundary bits then set them 
    q3.u.y = identity ;  
    q3.u.z = ( q3.u.z & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; 

    // HMM: can identity spare a bit for orient : as that would be cleaner ?
}
void quad4::get_prd( unsigned& boundary, unsigned& identity, float& orient )
{
    boundary = q3.u.x >> 16 ; 
    identity = q3.u.y ;  
    orient = ( q3.u.z & 0x80000000u ) ? -1.f : 1.f ;   
}

void quad4::set_flag( unsigned flag )
{
    q3.u.x = ( q3.u.x & 0xffff0000u ) | ( flag & 0xffffu ) ;   // clear flag bits then set them  
    q3.u.w |= flag ;    // bitwise-OR into flagmask 
}
void quad4::get_flag( unsigned& flag ) const 
{
    flag = q3.u.x & 0xffff ;  
}

unsigned quad4::flagmask() const 
{
    return q3.u.w ; 
}

void quad4::set_wavelength( float wl ){ q2.f.w = wl ; }
float quad4::wavelength() const { return q2.f.w ; } 


void quad4::set_flags(unsigned boundary, unsigned identity, unsigned idx, unsigned flag, float orient ) 
{
    q3.u.x = ((boundary & 0xffffu ) << 16) | (flag & 0xffffu ) ;    // hmm boundary only needs 8bits 0xff really 
    q3.u.y = identity ;      // identity needs 32bit as already bit packed primIdx and instanceIdx
    q3.u.z = ( orient < 0.f ? 0x80000000u : 0u ) | ( idx & 0x7fffffffu ) ;   // ~30bit gets up to a billion, ~2 bits spare  
    q3.u.w |= flag ;         // OpticksPhoton.h flags up to  0x1 << 15 : bitwise combination needs 16 bits,  16 bits spare  
    // hmm: could swap the general purpose identity for sensor index when this is a hit ?
}

void quad4::get_flags(unsigned& boundary, unsigned& identity, unsigned& idx, unsigned& flag, float& orient ) const
{
    boundary = q3.u.x >> 16 ; 
    flag = q3.u.x & 0xffffu ;  
    identity = q3.u.y ; 
    idx = ( q3.u.z & 0x7fffffffu ) ;
    orient = ( q3.u.z & 0x80000000u ) ? -1.f : 1.f ;   
}


/**
photon
---------

**/

struct photon
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

    SQUAD_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
    SQUAD_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; } 

    SQUAD_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it 
    SQUAD_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 

    SQUAD_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; }
    SQUAD_METHOD unsigned boundary() const { return boundary_flag >> 16 ; }

    SQUAD_METHOD void     set_flag(unsigned flag) {         boundary_flag = ( boundary_flag & 0xffff0000u ) | ( flag & 0xffffu ) ; flagmask |= flag ;  } // clear flag bits then set them  
    SQUAD_METHOD void     set_boundary(unsigned boundary) { boundary_flag = ( boundary_flag & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ; }        // clear boundary bits then set them 


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
inline std::string photon::desc() const 
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





struct qphoton
{
    union 
    {
        quad4  q ; 
        photon p ; 
    }; 
};




/**
Full example of numpy reporting of photon and record flags and boundaries in QUDARap/tests/mock_propagate.py::

    from opticks.ana.hismask import HisMask   
    hm = HisMask()

    boundary_ = lambda p:p.view(np.uint32)[3,0] >> 16
    flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff
    identity_ = lambda p:p.view(np.uint32)[3,1]   
    idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
    orient_   = lambda p:p.view(np.uint32)[3,2] >> 31
    flagmask_ = lambda p:p.view(np.uint32)[3,3]
    flagdesc_ = lambda p:" %4d %10s id:%6d ori:%d idx:%6d %10s " % ( boundary_(p), hm.label(flag_(p)), identity_(p), orient_(p), idx_(p), hm.label( flagmask_(p) ))

    from opticks.CSG.CSGFoundry import CSGFoundry 
    cf = CSGFoundry.Load()
    bflagdesc_ = lambda p:"%s : %s " % ( flagdesc_(p), cf.bndnamedict[boundary_(p)] )

**/


struct quad6 
{ 
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ;
    quad q4 ;
    quad q5 ;

    SQUAD_METHOD void zero();

};

void quad6::zero() 
{
    q0.u.x = 0 ; q0.u.y = 0 ; q0.u.z = 0 ; q0.u.w = 0 ; 
    q1.u.x = 0 ; q1.u.y = 0 ; q1.u.z = 0 ; q1.u.w = 0 ; 
    q2.u.x = 0 ; q2.u.y = 0 ; q2.u.z = 0 ; q2.u.w = 0 ; 
    q3.u.x = 0 ; q3.u.y = 0 ; q3.u.z = 0 ; q3.u.w = 0 ; 
    q4.u.x = 0 ; q4.u.y = 0 ; q4.u.z = 0 ; q4.u.w = 0 ; 
    q5.u.x = 0 ; q5.u.y = 0 ; q5.u.z = 0 ; q5.u.w = 0 ; 
} 





#if defined(__CUDACC__) || defined(__CUDABE__)
#else


inline std::ostream& operator<<(std::ostream& os, const quad4& v)  
{
    os  
       << v.q0.f  
       << v.q1.f  
       << v.q2.f  
       << v.q3.f
       ;   
    return os; 
}


inline std::ostream& operator<<(std::ostream& os, const quad6& v)  
{
    os  
       << v.q0.f  
       << v.q1.f  
       << v.q2.f  
       << v.q3.f
       << v.q4.f
       << v.q5.f
       ;   
    return os; 
}


/**
qvals
------

qvals extracts a sequence of floats from an environment string without
specifying a delimiter between the floats instead simply the numerical 
digits and + - . are used to find the floats.   

**/

inline void qvals( std::vector<float>& vals, const char* key, const char* fallback, int num_expect )
{
    char* val = getenv(key);
    char* p = const_cast<char*>( val ? val : fallback ); 
    while (*p) 
    {   
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-' || *p == '.') vals.push_back(strtof(p, &p)) ; 
        else p++ ;
    }   
    if( num_expect > 0 ) assert( vals.size() == unsigned(num_expect) ); 
}

inline void qvals( std::vector<long>& vals, const char* key, const char* fallback, int num_expect )
{
    char* val = getenv(key);
    char* p = const_cast<char*>( val ? val : fallback ); 
    while (*p) 
    {   
        if( (*p >= '0' && *p <= '9') || *p == '+' || *p == '-' ) vals.push_back(strtol(p, &p, 10)) ; 
        else p++ ;
    }   
    if( num_expect > 0 ) assert( vals.size() == unsigned(num_expect) ); 
}


inline void qvals( float& v,   const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 1 ); 
    v = vals[0] ; 
}

inline float qenvfloat( const char* key, const char* fallback )
{
    float v ; 
    qvals(v, key, fallback);  
    return v ; 
}

inline int qenvint( const char* key, const char* fallback )
{
    char* val = getenv(key);
    char* p = const_cast<char*>( val ? val : fallback ); 
    return std::atoi(p); 
}





inline void qvals( float2& v,  const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 2 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
}
inline void qvals( float3& v,  const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 3 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
    v.z = vals[2] ; 
}

inline void qvals( float3& v0,  float3& v1, const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 6 ); 

    v0.x = vals[0] ; 
    v0.y = vals[1] ; 
    v0.z = vals[2] ; 

    v1.x = vals[3] ; 
    v1.y = vals[4] ; 
    v1.z = vals[5] ; 
}

inline void qvals( float4& v0,  float4& v1, const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 8 ); 

    v0.x = vals[0] ; 
    v0.y = vals[1] ; 
    v0.z = vals[2] ; 
    v0.w = vals[3] ; 

    v1.x = vals[4] ; 
    v1.y = vals[5] ; 
    v1.z = vals[6] ; 
    v1.w = vals[7] ; 
}

inline void qvals( std::vector<float4>& v, const char* key, const char* fallback, bool normalize_ )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, -1 );

    unsigned num_values = vals.size() ; 
    assert( num_values % 4 == 0 );   

    unsigned num_v = num_values/4 ; 
    v.resize( num_v ); 

    for(unsigned i=0 ; i < num_v ; i++)
    {
        float4 vec = make_float4( vals[i*4+0], vals[i*4+1], vals[i*4+2], vals[i*4+3] ); 
        float3* v3 = (float3*)&vec.x ; 
        if(normalize_) *v3 = normalize(*v3); 

        v[i].x = vec.x ; 
        v[i].y = vec.y ; 
        v[i].z = vec.z ; 
        v[i].w = vec.w ; 
    }
}

inline void qvals( std::vector<float3>& v, const char* key, const char* fallback, bool normalize_ )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, -1 );

    unsigned num_values = vals.size() ; 
    assert( num_values % 3 == 0 );   

    unsigned num_v = num_values/3 ; 
    v.resize( num_v ); 

    for(unsigned i=0 ; i < num_v ; i++)
    {
        float3 vec = make_float3( vals[i*3+0], vals[i*3+1], vals[i*3+2]) ; 
        if(normalize_) vec = normalize(vec); 

        v[i].x = vec.x ; 
        v[i].y = vec.y ; 
        v[i].z = vec.z ; 
    }
}


inline void qvals( float4& v,  const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 4 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
    v.z = vals[2] ; 
    v.w = vals[3] ; 
}

inline int qvals4( float& x, float& y, float& z, float& w,  const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 4 ); 

    x = vals[0] ; 
    y = vals[1] ; 
    z = vals[2] ; 
    w = vals[3] ; 

    return 0 ; 
}

inline int qvals3( float& x, float& y, float& z, const char* key, const char* fallback )
{
    std::vector<float> vals ; 
    qvals( vals, key, fallback, 3 ); 

    x = vals[0] ; 
    y = vals[1] ; 
    z = vals[2] ; 

    return 0 ; 
}



inline void qvals( int& v,   const char* key, const char* fallback )
{
    std::vector<long> vals ; 
    qvals( vals, key, fallback, 1 ); 
    v = vals[0] ; 
}
inline void qvals( int2& v,  const char* key, const char* fallback )
{
    std::vector<long> vals ; 
    qvals( vals, key, fallback, 2 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
}
inline void qvals( int3& v,  const char* key, const char* fallback )
{
    std::vector<long> vals ; 
    qvals( vals, key, fallback, 3 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
    v.z = vals[2] ; 
}
inline void qvals( int4& v,  const char* key, const char* fallback )
{
    std::vector<long> vals ; 
    qvals( vals, key, fallback, 4 ); 
    v.x = vals[0] ; 
    v.y = vals[1] ; 
    v.z = vals[2] ; 
    v.w = vals[3] ; 
}



inline std::string quad4::desc() const 
{
    std::stringstream ss ;
    ss 
        << " post " << q0.f << std::endl
        << " momw " << q1.f << std::endl
        << " polw " << q2.f << std::endl
        << " flag " << q3.i << std::endl
        ;
    std::string s = ss.str();
    return s ;
}

/**
quad4::ephoton
---------------

*ephoton* is used from qudarap/tests/QSimTest generate tests as the initial photon, 
which gets persisted to p0.npy 
The script qudarap/tests/ephoton.sh sets the envvars defining the photon 
depending on the TEST envvar. 
 
**/

inline void quad4::ephoton() 
{
    qvals( q0.f ,      "EPHOTON_POST" , "0,0,0,0" );                      // position, time
    qvals( q1.f, q2.f, "EPHOTON_MOMW_POLW", "1,0,0,1,0,1,0,500" );  // direction, weight,  polarization, wavelength 
    qvals( q3.i ,      "EPHOTON_FLAG", "0,0,0,0" );   
    normalize_mom_pol(); 
    transverse_mom_pol(); 
}

inline void quad4::normalize_mom_pol() 
{
    float3* mom = (float3*)&q1.f.x ;  
    float3* pol = (float3*)&q2.f.x ;

    *mom = normalize(*mom); 
    *pol = normalize(*pol); 
}

inline void quad4::transverse_mom_pol() 
{
    float3* mom = (float3*)&q1.f.x ;  
    float3* pol = (float3*)&q2.f.x ;
    float mom_pol = fabsf( dot(*mom, *pol)) ;  
    float eps = 1e-5 ; 
    bool is_transverse = mom_pol < eps ; 

    if(!is_transverse )
    {
        std::cout 
             << " quad4::transverse_mom_pol " 
             << " FAIL "
             << " mom " << *mom 
             << " pol " << *pol 
             << " mom_pol " << mom_pol 
             << " eps " << eps 
             << " is_transverse " << is_transverse
             << std::endl 
             ; 
    }
    assert(is_transverse); 
}

inline quad4 quad4::make_ephoton()  // static
{
    quad4 q ; 
    q.ephoton(); 
    return q ; 
}


inline void quad2::eprd()
{
   qvals( q0.f, "EPRD_NRMT", "-1,0,0,100" ); 
   qvals( q1.i, "EPRD_FLAG", "101,0,0,10" ); 
   float3* nrm = normal(); 
   *nrm = normalize( *nrm ); 
}

inline quad2 quad2::make_eprd()  // static
{
    quad2 prd ; 
    prd.eprd(); 
    return prd ; 
}


inline std::string quad2::desc() const 
{
    std::stringstream ss ;
    ss 
        << " nrmt " << q0.f  
        << " flag " << q1.i 
        ;
    std::string s = ss.str();
    return s ;
}


#endif 





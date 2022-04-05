#pragma once

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


    SUTIL_INLINE SUTIL_HOSTDEVICE void zero();
    SUTIL_INLINE SUTIL_HOSTDEVICE float* data() ;
    SUTIL_INLINE SUTIL_HOSTDEVICE const float* cdata() const ;

    SUTIL_INLINE SUTIL_HOSTDEVICE float3* normal() ;
    SUTIL_INLINE SUTIL_HOSTDEVICE const float3* normal() const ;


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

float*         quad2::data() {         return &q0.f.x ;  }
const float*   quad2::cdata() const  { return &q0.f.x ;  }
float3*        quad2::normal() {       return (float3*)&q0.f.x ;  }
const float3*  quad2::normal() const { return (float3*)&q0.f.x ;  }




 

struct quad4 
{ 
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ;

    SUTIL_INLINE SUTIL_HOSTDEVICE void zero();
    SUTIL_INLINE SUTIL_HOSTDEVICE float* data() ;
    SUTIL_INLINE SUTIL_HOSTDEVICE const float* cdata() const ;

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static quad4 make_ephoton(); 
    void ephoton() ; 
    void normalize_mom_pol(); 
    void transverse_mom_pol(); 
#endif
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


struct quad6 
{ 
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ;
    quad q4 ;
    quad q5 ;

    SUTIL_INLINE SUTIL_HOSTDEVICE void zero();

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





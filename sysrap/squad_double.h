#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SQUAD_METHOD __host__ __device__ __forceinline__
#else
#    define SQUAD_METHOD inline 
#endif


union dquad
{
    double4    f ; 
    longlong4  i ; 
    ulonglong4 u ; 
};



struct dquad4 
{ 
    dquad q0 ; 
    dquad q1 ; 
    dquad q2 ; 
    dquad q3 ;

    SQUAD_METHOD void zero();
    SQUAD_METHOD double* data() ;
    SQUAD_METHOD const double* cdata() const ;
};

SQUAD_METHOD void dquad4::zero() 
{
    q0.u.x = 0 ; q0.u.y = 0 ; q0.u.z = 0 ; q0.u.w = 0 ; 
    q1.u.x = 0 ; q1.u.y = 0 ; q1.u.z = 0 ; q1.u.w = 0 ; 
    q2.u.x = 0 ; q2.u.y = 0 ; q2.u.z = 0 ; q2.u.w = 0 ; 
    q3.u.x = 0 ; q3.u.y = 0 ; q3.u.z = 0 ; q3.u.w = 0 ; 
} 

SQUAD_METHOD double*       dquad4::data() {         return &q0.f.x ;  }
SQUAD_METHOD const double* dquad4::cdata() const  { return &q0.f.x ;  }



#if defined(__CUDACC__) || defined(__CUDABE__)
#else


inline std::ostream& operator<<(std::ostream& os, const dquad& q)  
{
    os  
       << "f " << q.f  
    //   << "i " << q.i  
    //   << "u " << q.u  
       ;   
    return os; 
}


inline std::ostream& operator<<(std::ostream& os, const dquad4& v)  
{
    os 
       << std::endl  
       << v.q0.f 
       << std::endl 
       << v.q1.f 
       << std::endl 
       << v.q2.f 
       << std::endl  
       << v.q3.f 
       << std::endl 
       ;   
    return os; 
}

#endif




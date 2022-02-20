#pragma once
#include "scuda.h"


struct U4 { 
  unsigned char x ; 
  unsigned char y ;
  unsigned char z ;
  unsigned char w ; 
}; 

union U4U { 
   U4 u4 ; 
   unsigned u ;  
};  




struct C4 { 
  char x ; 
  char y ;
  char z ;
  unsigned char w ; 
}; 

union C4U { 
   C4 c4 ; 
   unsigned u ;  
};  


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>



inline std::string C4U_desc( const C4U& c4u )
{
    std::stringstream ss ; 
    ss << " C4U ( " 
       << std::setw(5) << int(c4u.c4.x)  
       << std::setw(5) << int(c4u.c4.y)   
       << std::setw(5) << int(c4u.c4.z)   
       << std::setw(5) << unsigned(c4u.c4.w)
       << " ) "
       ;   
    std::string s = ss.str(); 
    return s ; 
}

inline std::string C4U_desc( unsigned u )
{
    C4U c4u ; 
    c4u.u = u ; 

    return C4U_desc( c4u ); 
}

inline std::string C4U_desc( const int4& gsid )
{
    std::stringstream ss ; 
    ss << " C4U ( " 
       << std::setw(5) << gsid.x  
       << std::setw(5) << gsid.y   
       << std::setw(5) << gsid.z   
       << std::setw(5) << gsid.w
       << " ) "
       ;   
    std::string s = ss.str(); 
    return s ; 
}



inline std::string C4U_name( const int4& gsid, const char* prefix, char delim )
{
    std::stringstream ss ; 
    ss << prefix 
       << delim
       << gsid.x << delim
       << gsid.y << delim  
       << gsid.z << delim  
       << gsid.w
       ;   
    std::string s = ss.str(); 
    return s ; 
}
inline void C4U_decode( int4& gsid,  unsigned u )
{
    C4U c4u ; 
    c4u.u = u ; 

    gsid.x = int(c4u.c4.x) ; 
    gsid.y = int(c4u.c4.y) ; 
    gsid.z = int(c4u.c4.z) ; 
    gsid.w = int(c4u.c4.w) ; 
}


inline std::string C4U_name( unsigned gsid_ , const char* prefix, char delim )
{
    int4 gsid ; 
    C4U_decode( gsid, gsid_ );  
    return C4U_name( gsid, prefix, delim ); 
}
    



#endif 


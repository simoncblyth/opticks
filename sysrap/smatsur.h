#pragma once
/**
smatsur.h : "ems" : enumeration of Material and Surface types   
===============================================================

**/

enum {
    smatsur_Material                       = 0, 
    smatsur_NoSurface                      = 1,
    smatsur_Surface                        = 2,
    smatsur_Surface_zplus_sensor_A         = 3,
    smatsur_Surface_zplus_sensor_CustomART = 4,
    smatsur_Surface_zminus                 = 5
 
};
 
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>
#include <cassert>
#include <string>
#include <sstream>
#include <iomanip>
#include <csignal>

struct smatsur
{
    static constexpr const char* Material                        = "Material" ; 
    static constexpr const char* NoSurface                       = "NoSurface" ; 
    static constexpr const char* Surface                         = "Surface" ; 
    static constexpr const char* Surface_zplus_sensor_A          = "Surface_zplus_sensor_A" ; 
    static constexpr const char* Surface_zplus_sensor_CustomART  = "Surface_zplus_sensor_CustomART" ; 
    static constexpr const char* Surface_zminus                  = "Surface_zminus" ; 

    static int TypeFromChar(char OpticalSurfaceName0); 
    static std::string Desc(); 
    static int Type(const char* name); 
    static const char* Name(int type); 
};


inline int smatsur::TypeFromChar(char OpticalSurfaceName0)
{
    int type = -1  ;
    switch(OpticalSurfaceName0)
    {
        case '\0': type = smatsur_Material                       ; break ;  
        case '-':  type = smatsur_NoSurface                      ; break ;  
        case '@':  type = smatsur_Surface_zplus_sensor_CustomART ; break ; 
        case '#':  type = smatsur_Surface_zplus_sensor_A         ; break ; 
        case '!':  type = smatsur_Surface_zminus                 ; break ; 
        default:   type = smatsur_Surface                        ; break ; 
    }
    return type ; 
}

inline std::string smatsur::Desc() 
{
    const int N = 6 ; 
    char cc[N] = { '\0', '-', 'X', '#', '@', '!' } ; 
    std::stringstream ss ; 
    ss << "smatsur::Desc" << std::endl ; 
    for(int i=0 ; i < N ; i++)
    {
       char c = cc[i] ;  
       int type = TypeFromChar(c) ; 
       const char* name = Name(type) ;  
       int type2 = Type(name) ; 
       bool type_expect = type == type2 ;
       if(!type_expect) std::raise(SIGINT); 
       assert( type_expect ); 
       ss 
           << " c " << std::setw(3) << ( c == '\0' ? '0' : c ) 
           << " type " << std::setw(2) << type
           << " name " << name 
           << std::endl 
           ;
    }
    std::string str = ss.str(); 
    return str ; 
}

inline int smatsur::Type(const char* name)
{
    int type = -1  ;
    if(strcmp(name,Material)==0)                       type = smatsur_Material ;
    if(strcmp(name,NoSurface)==0)                      type = smatsur_NoSurface ;
    if(strcmp(name,Surface)==0)                        type = smatsur_Surface ;
    if(strcmp(name,Surface_zplus_sensor_A)==0)         type = smatsur_Surface_zplus_sensor_A ;
    if(strcmp(name,Surface_zplus_sensor_CustomART)==0) type = smatsur_Surface_zplus_sensor_CustomART ;
    if(strcmp(name,Surface_zminus)==0)                 type = smatsur_Surface_zminus ;
    return type ; 
}

inline const char* smatsur::Name(int type)
{
    const char* n = nullptr ;
    switch(type)
    {
        case smatsur_Material:                        n = Material                       ; break ;
        case smatsur_NoSurface:                       n = NoSurface                      ; break ;
        case smatsur_Surface:                         n = Surface                        ; break ;
        case smatsur_Surface_zplus_sensor_A:          n = Surface_zplus_sensor_A         ; break ;
        case smatsur_Surface_zplus_sensor_CustomART:  n = Surface_zplus_sensor_CustomART ; break ;
        case smatsur_Surface_zminus:                  n = Surface_zminus                 ; break ;
    }
    return n ; 
}

#endif


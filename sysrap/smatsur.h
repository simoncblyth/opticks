#pragma once
/**
smatsur.h : "ems" : enumeration of Material and Surface types   
===============================================================

**/

enum {
    smatsur_Material = 0, 
    smatsur_Surface  = 1,
    smatsur_Surface_zplus_sensor_A = 2,
    smatsur_Surface_zplus_sensor_CustomART = 3 
};
 
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cstring>

struct smatsur
{
    static constexpr const char* Material = "Material" ; 
    static constexpr const char* Surface  = "Surface" ; 
    static constexpr const char* Surface_zplus_sensor_A  = "Surface_zplus_sensor_A" ; 
    static constexpr const char* Surface_zplus_sensor_CustomART  = "Surface_zplus_sensor_CustomART" ; 

    static int Type(const char* name); 
    static const char* Name(int type); 
};


inline int smatsur::Type(const char* name)
{
    int type = -1  ;
    if(strcmp(name,Material)==0)                       type = smatsur_Material ;
    if(strcmp(name,Surface)==0)                        type = smatsur_Surface ;
    if(strcmp(name,Surface_zplus_sensor_A)==0)         type = smatsur_Surface_zplus_sensor_A ;
    if(strcmp(name,Surface_zplus_sensor_CustomART)==0) type = smatsur_Surface_zplus_sensor_CustomART ;
    return type ; 
}

inline const char* smatsur::Name(int type)
{
    const char* n = nullptr ;
    switch(type)
    {
        case smatsur_Material:                        n = Material                       ; break ;
        case smatsur_Surface:                         n = Surface                        ; break ;
        case smatsur_Surface_zplus_sensor_A:          n = Surface_zplus_sensor_A         ; break ;
        case smatsur_Surface_zplus_sensor_CustomART:  n = Surface_zplus_sensor_CustomART ; break ;
    }
    return n ; 
}

#endif


#pragma once
/**
SGLM_View.h
============

This is used from SGLM.h and SGLM_InterpolatedView.h


**/
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "NP.hh"

struct SGLM_View
{
    glm::tvec4<float> EYE ;
    glm::tvec4<float> LOOK ;
    glm::tvec4<float> UP ;
    glm::tvec4<unsigned> CTRL ;

    glm::quat getQuat() const ;
    std::string getELU() const ;
    std::string brief() const ;
    std::string desc() const ;
    bool is_zero() const ;
    void set_zero();


    template<typename T>
    static std::string FormatArray(const T* tt, int num, int precision, int width, char delim );

    template<typename T>
    static bool IsZero(const T* tt, int num);


};

inline glm::quat SGLM_View::getQuat() const
{
    return glm::quat_cast(glm::lookAt(glm::vec3(EYE), glm::vec3(LOOK), glm::vec3(UP)));
}

inline std::string SGLM_View::getELU() const
{
    int num = 3 ;
    int precision = 3 ;
    int width = 0 ;
    char delim = ',' ;

    std::stringstream ss ;
    ss
       << "ELU="
       << FormatArray<float>( glm::value_ptr(EYE), num, precision, width, delim )
       << ','
       << FormatArray<float>( glm::value_ptr(LOOK), num, precision, width, delim )
       << ','
       << FormatArray<float>( glm::value_ptr(UP), num, precision, width, delim )
       ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string SGLM_View::brief() const
{
    return getELU();
}

inline std::string SGLM_View::desc() const
{
    std::stringstream ss ;
    ss
        << " EYE  " << FormatArray<float>( glm::value_ptr(EYE),     4, 3, 10, '\0' )
        << " LOOK " << FormatArray<float>( glm::value_ptr(LOOK),    4, 3, 10, '\0' )
        << " UP   " << FormatArray<float>( glm::value_ptr(UP),      4, 3, 10, '\0' )
        << " CTRL " << FormatArray<unsigned>( glm::value_ptr(CTRL), 4, -1, 10, '\0' )
        << " is_zero " << ( is_zero() ? "YES" : "NO " )
        ;
    std::string str = ss.str();
    return str ;
}



bool SGLM_View::is_zero() const
{
    return
        IsZero<float>( glm::value_ptr(EYE), 4 ) &&
        IsZero<float>( glm::value_ptr(LOOK), 4 ) &&
        IsZero<float>( glm::value_ptr(UP), 4 ) &&
        IsZero<unsigned>( glm::value_ptr(CTRL), 4 ) &&
        true ;

}

void SGLM_View::set_zero()
{
    EYE.x = 0 ;
    EYE.y = 0 ;
    EYE.z = 0 ;
    EYE.w = 0 ;

    LOOK.x = 0 ;
    LOOK.y = 0 ;
    LOOK.z = 0 ;
    LOOK.w = 0 ;

    UP.x = 0 ;
    UP.y = 0 ;
    UP.z = 0 ;
    UP.w = 0 ;

    CTRL.x = 0 ;
    CTRL.y = 0 ;
    CTRL.z = 0 ;
    CTRL.w = 0 ;
}


template<typename T>
std::string SGLM_View::FormatArray(const T* tt, int num, int precision, int width, char delim )
{
    std::stringstream ss;
    for(int i=0; i < num; i++) {
        if(width > 0) ss << " " << std::setw(width);
        if(precision >= 0) ss << std::fixed << std::setprecision(precision);
        ss << tt[i];
        if(delim != '\0' && i < num - 1 ) ss << delim ;
    }
    return ss.str();
}

template<typename T>
bool SGLM_View::IsZero(const T* tt, int num )
{
    const T zero(0) ;
    int count = 0 ;
    for(int i=0; i < num; i++) if( tt[i] == zero ) count += 1 ;
    return count == num ;
}




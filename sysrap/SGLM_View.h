#pragma once
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
    std::string desc() const ;

    template<typename T>
    static std::string FormatArray(const T* tt, int num, int precision );
};

inline glm::quat SGLM_View::getQuat() const
{
    return glm::quat_cast(glm::lookAt(glm::vec3(EYE), glm::vec3(LOOK), glm::vec3(UP)));
}


inline std::string SGLM_View::desc() const
{
    std::stringstream ss ;
    ss
        << " EYE  " << FormatArray<float>( glm::value_ptr(EYE),     4, 3 )
        << " LOOK " << FormatArray<float>( glm::value_ptr(LOOK),    4, 3 )
        << " UP   " << FormatArray<float>( glm::value_ptr(UP),      4, 3 )
        << " CTRL " << FormatArray<unsigned>( glm::value_ptr(CTRL), 4, -1 )
        ;
    std::string str = ss.str();
    return str ;
}


template<typename T>
std::string SGLM_View::FormatArray(const T* tt, int num, int precision )
{
    std::stringstream ss;
    for(int i=0; i < num; i++) {
        ss << " " << std::setw(10);
        if(precision >= 0) ss << std::fixed << std::setprecision(precision);
        ss << tt[i];
    }
    return ss.str();
}



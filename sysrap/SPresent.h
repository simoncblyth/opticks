#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "scuda.h"


template<typename T>
inline std::string SPresent(const T* vv, unsigned n, const char* label, const char* mode )
{
    int lw = 15 ; 
    int w = 10 ; 
    std::stringstream ss ; 
    if(label) ss << std::setw(lw) << label << " "  ;
    if(mode)  ss <<  std::setw(lw) << mode << " "  ;

    for(unsigned i=0 ; i < n ; i++) 
    {
        if( n > 4 && i % 4 == 0 ) 
        {
             ss << std::endl << std::setw(lw) << ""  ; 
             if(mode) ss <<  std::setw(lw) << "" << " "  ;
        }
        ss << std::setw(w) << std::fixed << std::setprecision(3) << *(vv+i) ; 
    }
    ss << " " ; 
    std::string str = ss.str(); 
    return str ;  
}

template<typename T> std::string SPresent(const glm::tmat4x4<T>& m, const char* label, const char* mode=nullptr ){ return SPresent(glm::value_ptr(m), 16, label, mode); }
template<typename T> std::string SPresent(const glm::tvec4<T>&   v, const char* label, const char* mode=nullptr ){ return SPresent(glm::value_ptr(v),  4, label, mode); }
template<typename T> std::string SPresent(const glm::tvec3<T>&   v, const char* label, const char* mode=nullptr ){ return SPresent(glm::value_ptr(v),  3, label, mode); }

template<typename T>
inline std::string SPresent(std::vector<glm::tvec4<T>>& vv )
{
    int num_v = vv.size(); 
    std::stringstream ss ; 
    for(int i=0 ; i < num_v ; i++) ss << SPresent<T>(vv[i], nullptr) << std::endl ; 
    std::string str = ss.str(); 
    return str ;  
}

inline std::string SPresent(std::vector<float4>& vv )
{
    int num_v = vv.size(); 
    std::stringstream ss ; 
    for(int i=0 ; i < num_v ; i++) ss << vv[i] << std::endl ; 
    std::string str = ss.str(); 
    return str ;  
}




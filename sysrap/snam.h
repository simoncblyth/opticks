#pragma once

#include <vector>
#include <string>
#include <sstream>

struct snam
{
    static constexpr const char* WAVELENGTH = "wavelength.npy" ;
    static constexpr const char* ENERGY = "energy.npy" ;
    static constexpr const char* RAYLEIGH = "rayleigh.npy" ;
    static constexpr const char* MAT = "mat.npy" ;
    static constexpr const char* SUR = "sur.npy" ;
    static constexpr const char* BD = "bd.npy" ;
    static constexpr const char* BND = "bnd.npy" ;
    static constexpr const char* OPTICAL = "optical.npy" ;
    static constexpr const char* ICDF = "icdf.npy" ;

    static constexpr const char* MULTIFILM = "multifilm.npy" ;
    static constexpr const char* PROPCOM = "propcom.npy" ;

    static const char* get(const std::vector<std::string>& names, int idx ) ;  
    static std::string Desc(const std::vector<std::string>& names); 
};

inline const char* snam::get(const std::vector<std::string>& names, int idx)
{
    return idx > -1 && idx < int(names.size()) ? names[idx].c_str() : nullptr ; 
}

inline std::string snam::Desc(const std::vector<std::string>& names)
{
    std::stringstream ss ; 
    ss << "[snam::Desc names.size " << names.size() << "\n" ; 
    for(int i=0 ; i < int(names.size()) ; i++) ss << "[" << names[i] << "]\n" ; 
    ss << "]snam::Desc\n" ; 
    std::string str = ss.str() ; 
    return str ;  
}


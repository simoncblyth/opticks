#pragma once

#include <vector>
#include <string>

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
};

inline const char* snam::get(const std::vector<std::string>& names, int idx)
{
    return idx > -1 && idx < int(names.size()) ? names[idx].c_str() : nullptr ; 
}


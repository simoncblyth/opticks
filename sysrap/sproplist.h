#pragma once
/**
sproplist.h
===================

For MATERIAL the property default constants 
are taken from  GMaterialLib::defineDefaults

For SURFACE setting the prop values::

    (detect, absorb, reflect_specular, reflect_diffuse

requires access to optical surface type, 
if not already present need to add metadata 
to the surface NPFold/NP to carry that info.   

**/

#include "sprop.h"

struct sproplist
{
    static constexpr const char* MATERIAL = R"(
    0 0 RINDEX          1
    0 1 ABSLENGTH       1e6
    0 2 RAYLEIGH        1e6
    0 3 REEMISSIONPROB  0.
    1 0 GROUPVEL        299.792458
    1 1 SPARE11         0.
    1 2 SPARE12         0.
    1 3 SPARE13         0.
    )" ;
    // default GROUPVEL set to c_light_mm_per_ns, see U4PhysicalConstants.h 

    static constexpr const char* SURFACE = R"(
    0 0 EFFICIENCY      -2
    0 1 SPARE01         -2
    0 2 REFLECTIVITY    -2
    0 3 SPARE03         -2
    1 0 SPARE10         -2
    1 1 SPARE11         -2
    1 2 SPARE12         -2
    1 3 SPARE13         -2
    )" ;   

    static const sproplist* Material() ; 
    static const sproplist* Surface() ; 

    std::vector<sprop> PROP ; 
    sproplist(const char* spec ); 

    std::string desc() const ; 
    void getNames(std::vector<std::string>& pnames, const char* skip_prefix="SPARE") const ; 
    const sprop* findProp(const char* pname) const ; 
    const sprop* get(int g, int p) const ; 
};

inline const sproplist* sproplist::Material() // static
{
    return new sproplist(MATERIAL) ; 
}
inline const sproplist* sproplist::Surface() // static
{
    return new sproplist(SURFACE) ; 
}

inline sproplist::sproplist(const char* spec)
{
    sprop::Parse(PROP, spec); 
}
inline std::string sproplist::desc() const 
{
    return sprop::Desc(PROP); 
}
inline void sproplist::getNames(std::vector<std::string>& pnames, const char* skip_prefix ) const 
{
    sprop::GetNames(pnames, PROP, skip_prefix);  
}
inline const sprop* sproplist::findProp(const char* pname) const 
{
    return sprop::FindProp(PROP, pname); 
}
inline const sprop* sproplist::get(int g, int v) const 
{
    return sprop::Find(PROP, g, v) ; 
}



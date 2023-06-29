#pragma once
/**
sprop.h
=========

Curiously when using a constexpr std::array for the PROP::

    static constexpr std::array<sprop, 8> PROP = 
    {{
        { 0,0,"RINDEX" },
        { 0,1,"ABSLENGTH" },
        { 0,2,"RAYLEIGH" },
        { 0,3,"REEMISSIONPROB" },
        { 1,0,"GROUPVEL" },
        { 1,1,"SPARE11"  },
        { 1,2,"SPARE12"  },
        { 1,3,"SPARE13"  },
    }};

Find that had to do the below in the cc to avoid link errors::

    constexpr std::array<sprop,8> sprop_Material::PROP ;  

To avoid that have rejigged to extract the PROP from 
a multiline constexpr string instead of the array.  
**/

#include <string>
#include <vector>
#include <array>
#include "sstr.h"

struct sprop
{
    int  group ; 
    int  prop ;  
    char name[16] ; 
    std::string desc() const ; 
    bool parse(const char* str) ; 
    bool match(int g, int v) const ; 

    static void Parse(std::vector<sprop>& prop, const char* lines ); 
    static std::string Desc(const std::vector<sprop>& prop); 
    static void GetNames(std::vector<std::string>& pnames, const std::vector<sprop>& prop, const char* skip_prefix ); 
    static const sprop* FindProp(const std::vector<sprop>& prop, const char* pname ); 
    static const sprop* Find(const std::vector<sprop>& prop, int g, int v); 
}; 

inline std::string sprop::desc() const 
{
    std::stringstream ss ; 
    ss << "(" << group << "," << prop << ") " << name ; 
    std::string s = ss.str(); 
    return s ; 
}
inline bool sprop::parse(const char* str) 
{
    std::stringstream ss(str) ; 
    ss >> group >> prop >> name ; 
    if (ss.fail()) return false ; 
    return true ; 
}
inline bool sprop::match(int g, int v) const 
{
    return group == g && prop == v ; 
}

inline void sprop::Parse(std::vector<sprop>& prop, const char* lines ) // static 
{
    std::stringstream ss;  
    ss.str(lines) ;
    std::string s;
    while (std::getline(ss, s, '\n'))
    {
        const char* line = s.c_str() ; 
        sprop p ; 
        bool ok = p.parse(line) ; 
        //std::cout << "[" << line << "]" << " " << ( ok ? p.desc() : "FAIL" ) << std::endl ; 
        if(ok) prop.push_back(p) ; 
    }
}
inline std::string sprop::Desc(const std::vector<sprop>& prop)  // static 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < prop.size() ; i++) ss << prop[i].desc() << std::endl; 
    std::string s = ss.str(); 
    return s ; 
}
inline void sprop::GetNames(std::vector<std::string>& pnames, const std::vector<sprop>& prop, const char* skip_prefix )  // static
{
    for(unsigned i=0 ; i < prop.size() ; i++) 
    {
        const char* name = prop[i].name ; 
        if(sstr::MatchStart(name, skip_prefix) == false ) pnames.push_back(name) ; 
    }
}
inline const sprop* sprop::FindProp(const std::vector<sprop>& prop, const char* pname)
{
    const sprop* p = nullptr ; 
    for(unsigned i=0 ; i < prop.size() ; i++) if(strcmp(prop[i].name, pname)==0) p = &prop[i] ; 
    return p ; 
}
inline const sprop* sprop::Find(const std::vector<sprop>& prop, int g, int v)
{
    const sprop* p = nullptr ; 
    for(unsigned i=0 ; i < prop.size() ; i++) if(prop[i].match(g,v)) p = &prop[i] ; 
    return p ; 
}



struct sprop_Material
{
    enum { NUM_PAYLOAD_GRP = 2, NUM_PAYLOAD_VAL = 4 } ;

    static constexpr const char* SPEC = R"(
    0 0 RINDEX
    0 1 ABSLENGTH
    0 2 RAYLEIGH
    0 3 REEMISSIONPROB
    1 0 GROUPVEL
    1 1 SPARE11
    1 2 SPARE12
    1 3 SPARE13
    )" ;

    std::vector<sprop> PROP ; 

    sprop_Material(); 

    std::string desc() const ; 
    void getNames(std::vector<std::string>& pnames, const char* skip_prefix="SPARE") const ; 
    const sprop* findProp(const char* pname) const ; 
    const sprop* get(int g, int p) const ; 
};

inline sprop_Material::sprop_Material()
{
    sprop::Parse(PROP, SPEC); 
}
inline std::string sprop_Material::desc() const 
{
    return sprop::Desc(PROP); 
}
inline void sprop_Material::getNames(std::vector<std::string>& pnames, const char* skip_prefix ) const 
{
    sprop::GetNames(pnames, PROP, skip_prefix);  
}
inline const sprop* sprop_Material::findProp(const char* pname) const 
{
    return sprop::FindProp(PROP, pname); 
}
inline const sprop* sprop_Material::get(int g, int v) const 
{
    assert( g > -1 && g < NUM_PAYLOAD_GRP ); 
    assert( v > -1 && v < NUM_PAYLOAD_VAL ); 
    return sprop::Find(PROP, g, v) ; 
}






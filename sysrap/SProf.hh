#pragma once

#include <vector>
#include <fstream>
#include "sprof.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SProf
{
    static std::vector<sprof>       PROF ; 
    static std::vector<std::string> NAME ; 

    static void Add(const char* name); 
    static void Add( const char* name, const sprof& prof); 
    static void Clear(); 

    static std::string Serialize() ; 
    static std::string Desc() ; 

    static void Write(const char* path, bool append); 
    static void Read( const char* path ); 
};

inline void SProf::Add(const char* name)
{ 
    sprof prof ; 
    sprof::Stamp(prof); 
    Add(name, prof); 
}

inline void SProf::Add( const char* name, const sprof& prof)
{
    NAME.push_back(name); 
    PROF.push_back(prof); 
}

inline void SProf::Clear()
{
    PROF.clear(); 
    NAME.clear(); 
}

inline std::string SProf::Desc() // static
{
    int num_prof = PROF.size() ; 

    std::stringstream ss ; 
    for(int i=0 ; i < num_prof  ; i++) ss
        << std::setw(30) << NAME[i] 
        << " : " 
        << std::setw(50) << sprof::Desc_(PROF[i]) 
        << std::endl
        ; 

    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SProf::Serialize() // static
{
    int num = PROF.size() ; 
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++) ss << NAME[i] << ":" << sprof::Serialize(PROF[i]) << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline void SProf::Write(const char* path, bool append)
{
    std::ios_base::openmode mode = std::ios::out|std::ios::binary ; 
    if(append) mode |= std::ios::app ;
    std::ofstream fp(path, mode );
    fp << Serialize() ; 
    fp.close(); 
}

inline void SProf::Read(const char* path )
{
    Clear(); 
    bool dump = false ; 
    std::ifstream fp(path);

    if(fp.fail()) std::cerr 
        << "SProf::Read failed to read from" 
        << " [" << ( path ? path : "-" )  << "]" 
        << std::endl 
        ; 

    char delim = ':' ; 
    std::string str;
    while (std::getline(fp, str)) 
    {
        if(dump) std::cout << "str[" << str << "]" << std::endl ; 
        char* s = (char*)str.c_str() ; 
        char* p = (char*)strchr(s, delim ); 
        if(p == nullptr) continue ; 
        *p = '\0' ;  
        std::string name = s ; 
        sprof prof ; 
        int rc = sprof::Import(prof, p+1 );    
        if( rc == 0 ) Add( name.c_str(), prof ); 
        if(dump) std::cout << "name:[" << name << "]" << " Desc_:[" << sprof::Desc_(prof) << "]" << " rc " << rc  <<  std::endl ;  
    }
    fp.close(); 
}




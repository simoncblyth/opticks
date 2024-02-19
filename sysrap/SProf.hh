#pragma once
/**
SProf.hh
=========

HMM: collecting a full process lifecycle into SProf
is convenient, however have to avoid only saving 
at end of run : as with jobs prone to run out of 
memory want to have the run meta even when the 
job fails.   Hence better to write the run meta 
at the end of every event overwriting what was 
there before.

**/

#include <vector>
#include <fstream>
#include "sprof.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SProf
{
    static constexpr const char* FMT = "%0.3d" ; 
    static constexpr const int N = 10 ; 
    static char TAG[N] ; 
    static int  SetTag(int idx, const char* fmt=FMT ); 
    static bool HasTag(); 
    static const char* Tag(); 
    static void UnsetTag(); 

    static std::vector<sprof>       PROF ; 
    static std::vector<std::string> NAME ; 

    static void Add(const char* name); 
    static void Add( const char* name, const sprof& prof); 
    static int32_t Delta_RS();  // RS difference of last two stamps, or -1 when do not have more that one stamp 
    static int32_t Range_RS();  // RS range between first and last stamps  

    static void Clear(); 

    static std::string Serialize() ; 
    static std::string Desc() ; 

    static void Write(const char* path, bool append); 
    static void Read( const char* path ); 
};




inline int SProf::SetTag(int idx, const char* fmt)
{
    return snprintf(TAG, N, fmt, idx ); 
}
inline bool SProf::HasTag()
{   
    return TAG[0] != '\0' ; 
}
inline const char* SProf::Tag()
{
    return TAG[0] == '\0' ? nullptr : TAG ; 
}
inline void SProf::UnsetTag()
{
    TAG[0] = '\0' ; 
}





inline void SProf::Add(const char* name)
{ 
    sprof prof ; 
    sprof::Stamp(prof); 
    Add(name, prof); 
}

inline void SProf::Add( const char* _name, const sprof& prof)
{
    std::stringstream ss ; 
    ss << ( HasTag() ? TAG : "" ) << _name ;   
    std::string name = ss.str(); 
    NAME.push_back(name); 
    PROF.push_back(prof); 
}


inline int32_t SProf::Delta_RS()
{
    int num_prof = PROF.size(); 
    const sprof* p1 = num_prof > 0 ? &PROF.back() : nullptr ; 
    const sprof* p0 = num_prof > 1 ? p1 - 1 : nullptr ; 
    return p0 && p1 ? sprof::Delta_RS(p0, p1) : -1  ; 
}

inline int32_t SProf::Range_RS()
{
    int num_prof = PROF.size(); 
    const sprof* p0 = num_prof > 0 ? PROF.data() : nullptr ; 
    const sprof* p1 = num_prof > 0 ? PROF.data() + num_prof - 1 : nullptr ; 
    return p0 && p1 ? sprof::Delta_RS(p0, p1) : -1  ; 
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




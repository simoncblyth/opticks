#pragma once
/**
SProf.hh
=========

Collecting full process lifecycle into SProf is convenient
BUT: have to avoid only saving at end of run : as with jobs prone to run out of
memory want to have at least some of the profile data even when the job fails
Hence better to write the run meta at the end of every event
overwriting what was there before. The overwrite duplication is OK
as the expected number of profile stamps is small

KLUDGE::

    SProf::Write("run_meta.txt", true );
    // HMM: this relying on relative path, ie on the invoking directory


HMM Combining run metadata in run_meta.txt
(from SEvt::RUN_META) together with profiling info from here
is a kludge that is worth removing, because it complicates
things.

Best to keep things simple by:

1. write profiling info into "SProf.txt"
2. run meta into run_meta.txt

Actually the problem is not including metadata with profiling
its mixing across structs.


**/

#include <vector>
#include <fstream>
#include "sprof.h"
#include "ssys.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SProf
{
    static constexpr const char* SProf__WRITE = "SProf__WRITE" ;
    static constexpr const char* PATH = "SProf.txt" ;
    static constexpr const char* FMT = "%0.3d" ;
    static constexpr const int N = 10 ;
    static char TAG[N] ;
    static int  SetTag(int idx, const char* fmt=FMT ); // used to distinguish profiles from multiple events
    static bool HasTag();
    static const char* Tag();
    static void UnsetTag();

    static std::vector<sprof>       PROF ;   // sprof struct of 3 integers
    static std::vector<std::string> NAME ;
    static std::vector<std::string> META ;

    static int64_t Add(const char* name, const char* meta=nullptr);
    static void    Add(const char* name, const sprof& prof, const char* meta=nullptr );

    static int32_t Delta_RS();  // RS difference of last two stamps, or -1 when do not have more that one stamp
    static int32_t Range_RS();  // RS range between first and last stamps

    static void Clear();

    static std::string Serialize() ;
    static std::string Desc() ;

    static void Write(const char* path=PATH, bool append=false);
    static void Read( const char* path=PATH );
};


/**
SProf::SetTag
---------------

Canonically invoked from CSGOptiX::simulate

**/


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





inline int64_t SProf::Add(const char* name, const char* meta)
{
    sprof prof ;
    sprof::Stamp(prof);
    Add(name, prof, meta);
    return prof.st ;
}

inline void SProf::Add( const char* _name, const sprof& prof, const char* meta)
{
    std::stringstream ss ;
    ss << ( HasTag() ? TAG : "" ) << _name ;
    std::string name = ss.str();
    NAME.push_back(name);
    PROF.push_back(prof);
    META.push_back(meta ? meta : "");
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
    META.clear();
}

inline std::string SProf::Desc() // static
{
    int num_prof = PROF.size() ;

    std::stringstream ss ;
    for(int i=0 ; i < num_prof  ; i++)
    {
        const std::string& name = NAME[i] ;
        const sprof& prof = PROF[i] ;
        const std::string& meta = META[i] ;
        ss
            << std::setw(30) << name
            << " : "
            << std::setw(50) << sprof::Desc_(prof)
            << ( meta.empty() ? "" : " # " )
            << ( meta.empty() ? "" : meta  )
            << std::endl
            ;
    }

    std::string str = ss.str() ;
    return str ;
}

inline std::string SProf::Serialize() // static
{
    int num = PROF.size() ;
    std::stringstream ss ;
    for(int i=0 ; i < num ; i++)
    {
        const std::string& name = NAME[i] ;
        const sprof& prof = PROF[i] ;
        const std::string& meta = META[i] ;
        ss
            << name
            << ":"
            << sprof::Serialize(prof)
            << ( meta.empty() ? "" : " # " )
            << ( meta.empty() ? "" : meta  )
            << std::endl
            ;
    }
    std::string str = ss.str() ;
    return str ;
}

inline void SProf::Write(const char* path, bool append)
{
    bool WRITE = ssys::getenvbool(SProf__WRITE);
    if(!WRITE) std::cerr << "SProf::Write DISABLED, enable with [export " << SProf__WRITE << "=1]\n" ;
    if(!WRITE) return ;

    std::ios_base::openmode mode = std::ios::out|std::ios::binary ;
    if(append) mode |= std::ios::app ;
    std::ofstream fp(path, mode );
    fp << Serialize() ;
    fp.close();
}

/**

A000_QSim__simulate_HEAD:1760593677804471,7316440,1220224
A000_QSim__simulate_HEAD:1760593677804471,7316440,1220224   # metadata

**/


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

    char delim0 = ':' ;
    char delim1 = '#' ;

    std::string str;
    while (std::getline(fp, str))
    {
        if(dump) std::cout << "str[" << str << "]" << std::endl ;
        char* s = (char*)str.c_str() ;
        char* p = (char*)strchr(s, delim0 );
        char* h = (char*)strchr(s, delim1 );

        if(p == nullptr) continue ;
        *p = '\0' ;
        std::string name = s ;  // string prior to delim0 ":"

        bool with_meta = h != nullptr ;
        if(with_meta) *h = '\0' ;
        std::string meta = with_meta ? h + 2 : "" ;

        sprof prof ;
        int rc = sprof::Import(prof, p+1 );
        if( rc == 0 ) Add( name.c_str(), prof, meta.c_str() );
        if(dump) std::cout << "name:[" << name << "]" << " Desc_:[" << sprof::Desc_(prof) << "]" << " rc " << rc  <<  std::endl ;
    }
    fp.close();
}




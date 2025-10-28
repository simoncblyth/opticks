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


Usage
-------

::

    (ok) A[blyth@localhost CSGOptiX]$ opticks-f SProf::Write
    ./CSGOptiX/CSGOptiX.cc:    SProf::Write();
    ./qudarap/QSim.cc:    SProf::Write(); // per-event write, so have something in case of crash
    ./sysrap/SEvt.cc:    SProf::Write();
    ./sysrap/SEvt.cc:    SProf::Write();
    ./sysrap/SProf.hh:    SProf::Write("run_meta.txt", true );
    ./sysrap/SProf.hh:inline void SProf::Write(bool append)
    ./sysrap/SProf.hh:    if(!WRITE) std::cerr << "SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]\n" ;
    ./sysrap/tests/SProfTest.cc:        SProf::Write();      // frequent write to have something in case of crash
    ./sysrap/tests/SProfTest.cc:    SProf::Write();
    ./u4/tests/U4HitTest.cc:    SProf::Write(append);
    (ok) A[blyth@localhost opticks]$



Config via envvars
-------------------

Enable writing of profile txt file with::

    export SProf__WRITE=1

Default path to write profile info is "SProf.txt", to override that::

    export SProf__PATH=SProf_%0.5d.txt
    export SProf__PATH_INDEX=0

If the PATH provided contains "%" it is treated as a format string
which is expected to take one integer index provided from the PATH_INDEX.


Slurm array running without overwriting SProf.txt and other logs
-----------------------------------------------------------------

::

    LOGDIR=$SLURM_ARRAY_TASK_ID
    mkdir -p $LOGDIR
    cd $LOGDIR
    ...invoke executable...


Former Kludge, now removed
---------------------------

Formerly shared output from SProf into the run_meta.txt, using::

    SProf::Write("run_meta.txt", true );
    // HMM: this relying on relative path, ie on the invoking directory

HMM Combining run metadata in run_meta.txt
(from SEvt::RUN_META) together with profiling info from here
was a kludge that was removed as it complicates things.

**/

#include <vector>
#include <fstream>
#include "sprof.h"
#include "ssys.h"
#include "sstr.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SProf
{
    static constexpr const char* SProf__WRITE      = "SProf__WRITE" ;

    static constexpr const char* SProf__PATH       = "SProf__PATH" ;
    static constexpr const char* PATH_DEFAULT      = "SProf.txt" ;
    //static constexpr const char* PATH_DEFAULT    = "SProf_%0.5d.txt" ;

    static constexpr const char* SProf__PATH_INDEX = "SProf__PATH_INDEX" ;
    static constexpr const int PATH_INDEX_DEFAULT = 0 ;

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

    static const char* Path();
    static void Write(bool append=false);
    static void Read();
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


inline const char* SProf::Path()  // static
{
    const char* PATH = ssys::getenvvar(SProf__PATH,       PATH_DEFAULT);
    int INDEX        = ssys::getenvint(SProf__PATH_INDEX, PATH_INDEX_DEFAULT);
    bool looks_like_fmt = strstr(PATH,"%") != nullptr ;
    const char* path = looks_like_fmt ? sstr::Format(PATH, INDEX) : PATH ;
    return path ;
}

inline void SProf::Write(bool append)
{
    bool WRITE = ssys::getenvbool(SProf__WRITE) ;
    if(!WRITE) std::cerr << "SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]\n" ;
    if(!WRITE) return ;

    const char* path = Path();
    if(!path) return ;
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


inline void SProf::Read()
{
    const char* path = Path();
    if(!path) return ;

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




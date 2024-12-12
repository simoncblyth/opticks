#pragma once
/**
SCurandChunk.h  NB NOT GENERAL : THIS IS SPECIFIC TO curandStateXORWOW
========================================================================


The Load_ and Save methods are specific to curandStateXORWOW, most 
of the rest is more general. But there is no need 
for saving states for counter based RNG such as Philox 

::

    ~/o/sysrap/tests/SCurandState_test.sh

**/

#include <iomanip>

#include "sdirectory.h"
#include "spath.h"
#include "sstr.h"
#include "ssys.h"
#include "scurandref.h"
#include "sdigest.h"

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SCurandChunk
{
    typedef unsigned long long ULL ; 
    scurandref<curandStateXORWOW> ref = {} ; 

    std::string desc() const ;
    std::string meta() const ;
    std::string name() const ;
    const char* path(const char* _dir=nullptr) const ;
    bool path_exists(const char* _dir=nullptr) const ; 

    static constexpr const long STATE_SIZE = 44 ;  
    static constexpr const char* RNGDIR = "${RNGDir:-$HOME/.opticks/rngcache/RNG}" ; 
    static constexpr const ULL M = 1000000 ; 
    static constexpr const char* PREFIX = "SCurandChunk_" ; 
    static constexpr const char* EXT = ".bin" ; 
    static constexpr char DELIM = '_' ;
    static constexpr const long NUM_ELEM = 5 ;  

    static const char* Dir( const char* _dir=nullptr ); 
    static std::string Desc(const SCurandChunk& chunk, const char* _dir=nullptr ); 
    static std::string Desc(const std::vector<SCurandChunk>& chunk, const char* _dir=nullptr ); 
    static int ParseDir( std::vector<SCurandChunk>& chunk, const char* _dir=nullptr );

    static constexpr const char* ParseName_DEBUG = "SCurandChunk__ParseName_DEBUG" ; 
    static int ParseName( SCurandChunk& chunk, const char* name ); 
    static long ParseNum(const char* num); 

    static std::string FormatIdx(ULL idx);
    static std::string FormatNum(ULL num);
    static std::string FormatMeta(const scurandref<curandStateXORWOW>& d ); 
    static std::string FormatName(const scurandref<curandStateXORWOW>& d ); 

    static ULL NumFromFilesize(const char* name, const char* _dir=nullptr); 
    static bool IsValid(const SCurandChunk& chunk, const char* _dir=nullptr);
    static int CountValid(const std::vector<SCurandChunk>& chunk, const char* _dir=nullptr );
    static scurandref<curandStateXORWOW>* Find(std::vector<SCurandChunk>& chunk, long idx );
    static ULL NumTotal_SpecCheck(const std::vector<SCurandChunk>& chunk, const std::vector<ULL>& spec );
    static ULL NumTotal_InRange(  const std::vector<SCurandChunk>& chunk, ULL i0, ULL i1 ); 

    scurandref<curandStateXORWOW> load(ULL read_num=0, const char* _dir=nullptr, sdigest* dig=nullptr ) const ; 
    static int OldLoad( SCurandChunk& chunk, const char* name, ULL q_num=0, const char* _dir=nullptr );
    static curandStateXORWOW* Load_( ULL& file_num, const char* path, ULL read_num, sdigest* dig ); 

    static int Save( curandStateXORWOW* states, unsigned num_states, const char* path ) ; 
    int save( const char* _dir=nullptr ) const ; 
};

inline std::string SCurandChunk::desc() const
{
    std::stringstream ss ; 
    ss << Desc(*this) << "\n" ; 
    std::string str = ss.str() ; 
    return str ;  
}

inline std::string SCurandChunk::meta() const
{
    return FormatMeta(ref); 
}
inline std::string SCurandChunk::name() const
{
    return FormatName(ref); 
}
inline const char* SCurandChunk::path(const char* _dir) const
{
    std::string n = name(); 
    const char* dir = _dir ? _dir : RNGDIR ; 
    return spath::Resolve( dir, n.c_str() );  
}

inline bool SCurandChunk::path_exists(const char* _dir) const
{
    const char* pth = path(_dir) ; 
    return spath::Exists(pth); 
} 

inline const char* SCurandChunk::Dir( const char* _dir )
{
    const char* dir = _dir ? _dir : RNGDIR ; 
    return spath::Resolve(dir); 
} 

inline std::string SCurandChunk::Desc(const SCurandChunk& chunk, const char* _dir )
{
    bool exists = chunk.path_exists(_dir); 
    std::stringstream ss ; 
    ss << chunk.path(_dir) << " exists " <<  ( exists ? "YES" : "NO " ) ; 
    std::string str = ss.str() ; 
    return str ;  
}

inline std::string SCurandChunk::Desc(const std::vector<SCurandChunk>& chunk, const char* _dir )
{
    int num_chunk = chunk.size(); 
    std::stringstream ss ; 
    ss << "SCurandChunk::Desc\n" ; 
    for(int i=0 ; i < num_chunk ; i++) ss << Desc(chunk[i], _dir) << "\n" ;  
    std::string str = ss.str() ; 
    return str ;  
} 

/**
SCurandChunk::ParseDir
-----------------------

Populate chunk vector based on matching file names within directory 

**/


inline int SCurandChunk::ParseDir(std::vector<SCurandChunk>& chunk, const char* _dir )
{
    const char* dir = spath::Resolve(_dir ? _dir : RNGDIR) ; 
    std::vector<std::string> names ; 
    sdirectory::DirList( names, dir, PREFIX, EXT ); 

    int num_names = names.size(); 
    int count = 0 ; 

    for(int i=0 ; i < num_names ; i++) 
    {
        const std::string& n = names[i] ; 
        SCurandChunk c = {} ; 
        if(SCurandChunk::ParseName(c, n.c_str())==0) 
        {
            ULL chunk_offset = NumTotal_InRange(chunk, 0, chunk.size() ); 
            assert( c.ref.chunk_offset == chunk_offset );
            assert( c.ref.chunk_idx == count ); // chunk files must be in idx order 
            chunk.push_back(c);
            count += 1 ; 
        }
    }
    return 0 ; 
}

/**
SCurandChunk::ParseName
-------------------------

::

    SCurandChunk_0000_0000M_0001M_0_0.bin
    SCurandChunk_0001_0001M_0001M_0_0.bin
    SCurandChunk_0002_0002M_0001M_0_0.bin
    SCurandChunk_0003_0003M_0001M_0_0.bin
    SCurandChunk_0004_0004M_0001M_0_0.bin
    SCurandChunk_0005_0005M_0001M_0_0.bin
    ^^^^^^^^^^^^^....................^^^^
    PREFIX             meta          EXT

**/


inline int SCurandChunk::ParseName( SCurandChunk& chunk, const char* name )
{
    if(name == nullptr ) return 1 ;  
    size_t other = strlen(PREFIX)+strlen(EXT) ; 
    if( strlen(name) <= other ) return 2 ; 

    std::string n = name ; 
    std::string meta = n.substr( strlen(PREFIX), strlen(name) - other ); 

    std::vector<std::string> elem ; 
    sstr::Split(meta.c_str(), DELIM, elem ); 

    unsigned num_elem = elem.size(); 
    if( num_elem != NUM_ELEM )  return 3 ; 

    chunk.ref.chunk_idx    = std::atoll(elem[0].c_str()) ; 
    chunk.ref.chunk_offset = ParseNum(elem[1].c_str()) ;
    chunk.ref.num          = ParseNum(elem[2].c_str()) ; 
    chunk.ref.seed         = std::atoll(elem[3].c_str()) ; 
    chunk.ref.offset       = std::atoll(elem[4].c_str()) ; 
    chunk.ref.states       = nullptr ; 

    int DEBUG = ssys::getenvint(ParseName_DEBUG,0); 

    if(DEBUG > 0) std::cout 
         << ParseName_DEBUG  
         << " " << std::setw(30) << n 
         << " : [" << meta << "][" 
         << chunk.name() 
         << "]\n" 
         ; 
    return 0 ; 
}



inline long SCurandChunk::ParseNum(const char* num)
{
    char* n = strdup(num); 
    char last = n[strlen(n)-1] ; 
    ULL scale = last == 'M' ? M : 1 ; 
    if(scale > 1) n[strlen(n)-1] = '\0' ; 
    ULL value = scale*std::atoll(num) ; 
    return value ; 
}



inline std::string SCurandChunk::FormatMeta(const scurandref<curandStateXORWOW>& d)
{
    std::stringstream ss ; 
    ss 
       << FormatIdx(d.chunk_idx) 
       << DELIM
       << FormatNum(d.chunk_offset) 
       << DELIM
       << FormatNum(d.num)
       << DELIM
       << d.seed
       << DELIM
       << d.offset
       ; 
    std::string str = ss.str(); 
    return str ;   
}
inline std::string SCurandChunk::FormatName(const scurandref<curandStateXORWOW>& d)
{
    std::stringstream ss ; 
    ss << PREFIX << FormatMeta(d) << EXT ; 
    std::string str = ss.str(); 
    return str ;   
}
 
inline std::string SCurandChunk::FormatIdx(ULL idx)
{
    std::stringstream ss; 
    ss << std::setw(4) << std::setfill('0') << idx ;
    std::string str = ss.str(); 
    return str ;   
}
inline std::string SCurandChunk::FormatNum(ULL num)
{
    ULL scale = M  ; 

    bool intmul = num % scale == 0 ; 
    if(!intmul) std::cerr
         << "SCurandChunk::FormatNum"
         << " num [" << num << "]"
         << " intmul " << ( intmul ? "YES" : "NO " )
         << "\n"
         ; 
    assert( intmul && "integer multiples of 1000000 are required" ); 
    ULL value = num/scale ; 

    std::stringstream ss; 
    ss << std::setw(4) << std::setfill('0') << value << 'M' ; 
    std::string str = ss.str(); 
    return str ;   
}

inline unsigned long long SCurandChunk::NumFromFilesize(const char* name, const char* _dir)
{
    const char* dir = _dir ? _dir : RNGDIR ; 
    ULL file_size = spath::Filesize(dir, name); 

    bool expected_file_size = file_size % STATE_SIZE == 0 ; 
    ULL file_num = file_size/STATE_SIZE ;

    if(0) std::cerr
        << "SCurandChunk::NumFromFilesize"
        << " dir " << ( dir ? dir : "-" )
        << " name " << ( name ? name : "-" )
        << " file_size " << file_size
        << " STATE_SIZE " << STATE_SIZE
        << " file_num " << file_num
        << " expected_file_size " << ( expected_file_size ? "YES" : "NO" )
        << "\n"
        ;

    assert( expected_file_size );
    return file_num ; 
}

inline bool SCurandChunk::IsValid(const SCurandChunk& chunk, const char* _dir )
{
    bool exists = chunk.path_exists(_dir) ;  
    if(!exists) return false ; 

    std::string n = chunk.name(); 


    ULL chunk_num = chunk.ref.num ; 
    ULL file_num = NumFromFilesize(n.c_str(), _dir) ;  
    bool valid = chunk_num == file_num ; 

    if(!valid) std::cerr   
        << "SCurandChunk::IsValid"
        << " chunk file exists [" << n << "]"
        << " but filesize does not match name metadata "
        << " chunk_num " << chunk_num
        << " file_num " << file_num
        << " valid " << ( valid ? "YES" : "NO " )
        << "\n"
        ;
    return valid ; 
}

inline int SCurandChunk::CountValid(const std::vector<SCurandChunk>& chunk, const char* _dir )
{
    int num_chunk = chunk.size(); 
    int count = 0 ; 
    for(int i=0 ; i < num_chunk ; i++)
    {
        const SCurandChunk& c = chunk[i];     
        bool valid = IsValid(c, _dir); 
        if(!valid) continue ;  
        count += 1 ;          
    }
    return count ; 
}

inline scurandref<curandStateXORWOW>* SCurandChunk::Find(std::vector<SCurandChunk>& chunk, long q_idx )
{
    int num_chunk = chunk.size(); 
    int count = 0 ; 
    scurandref<curandStateXORWOW>* p = nullptr ; 
    for(int i=0 ; i < num_chunk ; i++)
    {
        SCurandChunk& c = chunk[i] ;     
        if( c.ref.chunk_idx == q_idx ) 
        {
            p = &(c.ref) ;  
            count += 1 ; 
        }
    }
    assert( count == 0 || count == 1 ); 
    return count == 1 ? p : nullptr ; 
}


/**
SCurandChunk::NumTotal_SpecCheck
---------------------------------

Total number of states in the chunks 

**/


inline unsigned long long SCurandChunk::NumTotal_SpecCheck(const std::vector<SCurandChunk>& chunk, const std::vector<ULL>& spec )
{
    assert( chunk.size() == spec.size() ) ;
    ULL tot = 0 ; 
    ULL num_chunk = chunk.size(); 
    for(ULL i=0 ; i < num_chunk ; i++)
    {
        const scurandref<curandStateXORWOW>& d = chunk[i].ref ;     
        assert( d.chunk_idx == i );  
        assert( d.num == spec[i] );  
        tot += d.num ; 
    }
    return tot ; 
}

inline unsigned long long SCurandChunk::NumTotal_InRange( const std::vector<SCurandChunk>& chunk, ULL i0, ULL i1 )
{
    ULL num_chunk = chunk.size(); 
    assert( i0 <= num_chunk ); 
    assert( i1 <= num_chunk ); 

    ULL num_tot = 0ull ; 
    for(ULL i=i0 ; i < i1 ; i++) 
    {
        const scurandref<curandStateXORWOW>& d = chunk[i].ref ;     
        num_tot += d.num ; 
    } 
    return num_tot ; 
}









inline scurandref<curandStateXORWOW> SCurandChunk::load( ULL read_num, const char* _dir, sdigest* dig ) const
{
    scurandref<curandStateXORWOW> lref(ref); 
    const char* p = path(_dir); 

    ULL file_num = 0 ; 
    lref.states = Load_(file_num, p, read_num, dig );
    lref.num = read_num ; 

    return lref ;
}


/**
SCurandChunk::OldLoad
---------------------

**/


inline int SCurandChunk::OldLoad( SCurandChunk& chunk, const char* name, ULL read_num, const char* _dir )
{
    int prc = ParseName(chunk, name); 
    if( prc > 0) std::cerr
        << "SCurandChunk::Load"
        << " chunk name not allowed "
        << "\n"
        ;
    if(prc > 0) return 1 ; 
    
    const char* dir = _dir ? _dir : RNGDIR ; 
    const char* p = spath::Resolve(dir, name); 

    ULL name_num = chunk.ref.num ;         // from ParseName

    ULL file_num = 0 ; 
    curandStateXORWOW* h_states = Load_( file_num, p, read_num, nullptr ) ; 

    if( h_states )
    {
        chunk.ref.num = read_num ; 
        chunk.ref.states = h_states ; 
        bool name_filesize_consistent = file_num == name_num ; 

        std::cerr
            << "SCurandChunk::Load"
            << " path " << p 
            << " name_num " << FormatNum(name_num) << "(from parsing filename) "
            << " file_num " << FormatNum(file_num) << "(from file_size/STATE_SIZE) "
            << " name_filesize_consistent " << ( name_filesize_consistent ? "YES" : "NO " )
            << " read_num " << FormatNum(read_num)
            << "\n"
            ; 
    }
    return h_states ? 0 : 2  ; 
}


inline curandStateXORWOW* SCurandChunk::Load_( ULL& file_num, const char* path, ULL read_num, sdigest* dig )
{
    FILE *fp = fopen(path,"rb");

    bool open_failed = fp == nullptr ; 
    if(open_failed) std::cerr 
        << "SCurandChunk::Load_"
        << " unable to open file "
        << "[" << path << "]" 
        << "\n" 
        ; 

    if(open_failed) return nullptr ; 

    fseek(fp, 0L, SEEK_END);
    ULL file_size = ftell(fp);
    rewind(fp);

    bool expected_size = file_size % STATE_SIZE == 0 ; 
    if(!expected_size) std::cerr 
        << "SCurandChunk::Load_"
        << " expected_size " << ( expected_size ? "YES" : "NO " )
        << "\n" 
        ;  

    if(!expected_size) return nullptr ; 

    file_num = file_size/STATE_SIZE;   // NB  STATE_SIZE not same as type_size

    bool read_num_toobig = read_num > file_num ; 
    if(read_num_toobig) std::cerr 
        << "SCurandChunk::Load_"
        << " read_num_toobig " << ( read_num_toobig ? "YES" : "NO " )
        << "\n" 
        ;  

    if(read_num_toobig) return nullptr ; 
    if(read_num == 0) read_num = file_num ;  // 0 means all 

    curandStateXORWOW* h_states = (curandStateXORWOW*)malloc(sizeof(curandStateXORWOW)*read_num);

/**
/usr/local/cuda-11.7/include/curand_kernel.h::

     140 struct curandStateXORWOW {
     141     unsigned int d, v[5];
     142     int boxmuller_flag;
     143     int boxmuller_flag_double;
     144     float boxmuller_extra;
     145     double boxmuller_extra_double;
     146 };


**/


    for(ULL i = 0 ; i < read_num ; ++i )
    {   
        curandStateXORWOW& rng = h_states[i] ;
        fread(&rng.d,                     sizeof(unsigned),1,fp);  if(dig) dig->add_<unsigned>(&rng.d, 1 ); 
        fread(&rng.v,                     sizeof(unsigned),5,fp);  if(dig) dig->add_<unsigned>( rng.v, 5 );
        fread(&rng.boxmuller_flag,        sizeof(int)     ,1,fp);  if(dig) dig->add_<int>(&rng.boxmuller_flag,1); 
        fread(&rng.boxmuller_flag_double, sizeof(int)     ,1,fp);  if(dig) dig->add_<int>(&rng.boxmuller_flag_double,1);
        fread(&rng.boxmuller_extra,       sizeof(float)   ,1,fp);  if(dig) dig->add_<float>(&rng.boxmuller_extra,1);
        fread(&rng.boxmuller_extra_double,sizeof(double)  ,1,fp);  if(dig) dig->add_<double>(&rng.boxmuller_extra_double,1);
    }   
    fclose(fp);
    return h_states ; 
}


inline int SCurandChunk::Save( curandStateXORWOW* states, unsigned num_states, const char* path ) // static
{
    sdirectory::MakeDirsForFile(path);
    FILE *fp = fopen(path,"wb");
    bool open_fail = fp == nullptr ; 

    if(open_fail) std::cerr 
        << "SCurandChunk::Save"
        << " FAILED to open file for writing" 
        << ( path ? path : "-" ) 
        << "\n" 
        ;

    if( open_fail ) return 3 ; 

    for(unsigned i = 0 ; i < num_states ; ++i )
    {
        curandStateXORWOW& rng = states[i] ;
        fwrite(&rng.d,                     sizeof(unsigned int),1,fp);
        fwrite(&rng.v,                     sizeof(unsigned int),5,fp);
        fwrite(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);
        fwrite(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);
    }   
    fclose(fp);
    return 0 ; 
}

inline int SCurandChunk::save( const char* _dir ) const
{
    const char* p = path(_dir);  
    bool exists = spath::Exists(p); 
    std::cerr 
        << "SCurandChunk::save"
        << " p " << ( p ? p : "-" )
        << " exists " << ( exists ? "YES" : "NO " )
        << "\n"
        ;  

    if(exists) return 1 ; 

    int rc = Save( ref.states, ref.num, p ); 
    return rc ;
}


#pragma once
/**
np.h
=====

https://github.com/simoncblyth/np/

Extract from NP.hh NPU.hh minimal-ish code to write a NumPy file

**/
#include <cassert>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

struct np
{
    static constexpr char* MAGIC = (char*)"\x93NUMPY" ; 
    static constexpr bool FORTRAN_ORDER = false ;
    static char constexpr ENDIAN_LITTLE = '<' ;
    static char constexpr ENDIAN_BIG = '>' ;

    static std::string _make_header(const std::vector<int>& shape, const char* descr="<f8" );
    static std::string _make_header(const std::string& dict); 
    static std::string _make_preamble( int major=1, int minor=0 );
    static std::string _little_endian_short_string( uint16_t dlen ) ; 
    static std::string _make_tuple(const std::vector<int>& shape, bool json );
    static std::string _make_dict(const std::vector<int>& shape, const char* descr );

    template<typename T>
    static void Write(const char* path, const std::vector<int>& shape, const T* data, const char* descr="<f8" ); 

    template<typename T>
    static void Write(const char* dir, const char* name, const std::vector<int>& shape, const T* data, const char* descr="<f8" ); 

    static std::string FormPath( const char* dir, const char* name ); 
    static void WriteString( const char* dir, const char* name, const char* txt );


}; 

// NPU::_make_header
inline std::string np::_make_header(const std::vector<int>& shape, const char* descr )
{
    std::string dict = _make_dict( shape, descr ); 
    std::string header = _make_header( dict ); 
    return header ; 
}

// NPU::_make_header
inline std::string np::_make_header(const std::string& dict)
{
    uint16_t dlen = dict.size() ;
    uint16_t padding = 16 - ((10 + dlen ) % 16 ) - 1 ;
    padding += 3*16 ; // adhoc extra padding for bit-perfect matching to NumPy (for test array anyhow)
    uint16_t hlen = dlen + padding + 1 ; 

    assert( (hlen + 10) % 16 == 0 );  
    std::stringstream ss ; 
    ss << _make_preamble() ;  
    ss << _little_endian_short_string( hlen ) ; 
    ss << dict ; 
 
    for(int i=0 ; i < padding ; i++ ) ss << " " ; 
    ss << "\n" ;  

    return ss.str(); 
}

// NPU::_make_preamble
inline std::string np::_make_preamble( int major, int minor )
{
    std::string preamble(MAGIC) ; 
    preamble.push_back((char)major); 
    preamble.push_back((char)minor); 
    return preamble ; 
}

// NPU::_little_endian_short_string
inline std::string np::_little_endian_short_string( uint16_t dlen )
{
    union u16c2_t { 
        uint16_t u16 ; 
        char     c[2] ;  
    }; 

    u16c2_t len ; 
    len.u16 = dlen ; 

    unsigned one = 1u ; 
    char e = *(char *)&one == 1 ? ENDIAN_LITTLE : ENDIAN_BIG ;   
    std::string hlen(2, ' ') ;
    hlen[0] = e == ENDIAN_LITTLE ? len.c[0] : len.c[1] ;  
    hlen[1] = e == ENDIAN_LITTLE ? len.c[1] : len.c[0] ; 
    return hlen ; 
}

// NPU::_make_tuple
inline std::string np::_make_tuple( const std::vector<int>& shape, bool json )
{
    int ndim = shape.size() ;
    std::stringstream ss ; 
    ss <<  ( json ? "[" : "(" ) ; 

    if( ndim == 1)
    {
        ss << shape[0] << "," ; 
    }
    else
    {
        for(int i=0 ; i < ndim ; i++ ) ss << shape[i] << ( i == ndim - 1 ? "" : ", " )  ; 
    }
    ss << ( json ?  "] " : "), " ) ;    // hmm assuming shape comes last in json
    return ss.str(); 
}

// NPU::_make_dict
inline std::string np::_make_dict(const std::vector<int>& shape, const char* descr )
{
    std::stringstream ss ; 
    ss << "{" ; 
    ss << "'descr': '" << descr << "', " ; 
    ss << "'fortran_order': " << ( FORTRAN_ORDER ? "True" : "False" ) << ", " ; 
    ss << "'shape': " ; 
    bool json = false ; 
    ss << _make_tuple( shape, json ) ; 
    ss << "}" ;  
    return ss.str(); 
} 



// adapt from NP::save
template<typename T>
inline void np::Write(const char* path, const std::vector<int>& shape, const T* data, const char* descr ) 
{
    int nv = 1 ; 
    for(int i=0 ; i < int(shape.size()) ; i++) nv *= shape[i] ; 
    int arr_bytes = sizeof(T)*nv ;  

    std::string hdr = _make_header(shape, descr); 
    std::ofstream fpa(path, std::ios::out|std::ios::binary);
    fpa << hdr ; 
    fpa.write( (char*)data, arr_bytes );
}

template<typename T>
inline void np::Write(const char* dir, const char* name, const std::vector<int>& shape, const T* data, const char* descr ) 
{
    std::string path = FormPath(dir, name); 
    Write<T>( path.c_str(), shape, data, descr );
}

inline std::string np::FormPath( const char* dir, const char* name )
{
    std::stringstream ss ; 
    if(dir) ss << dir << "/" ; 
    if(name) ss << name ;    
    std::string path = ss.str(); 
    return path ; 
}

inline void np::WriteString( const char* dir, const char* name, const char* txt )
{
    std::string path = FormPath(dir, name); 
    if(txt == nullptr) return ; 
    std::ofstream fp(path.c_str(), std::ios::out);
    fp << txt ; 
    fp.close(); 
}



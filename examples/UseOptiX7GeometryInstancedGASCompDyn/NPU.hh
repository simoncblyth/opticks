#pragma once

#include <sstream>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>
#include <complex>
#include <fstream>
#include <cstdlib>
#include <cstdint>


union u16c2_t 
{
   uint16_t u16 ; 
   char     c[2] ;  
}; 

struct Endian
{
    static const unsigned UNITY ; 
    static const char LITTLE ; 
    static const char BIG ; 
    static char detect(); 
};

const unsigned Endian::UNITY = 1 ; 
const char Endian::LITTLE = '<' ;  // least-significant byte at the smallest address
const char Endian::BIG = '>' ;     // most-significant byte at the smallest address

inline char Endian::detect()
{
    return (*(char *)&UNITY == 1) ? LITTLE : BIG ; 
}

template<typename T>
struct Desc 
{
    static const char code ; 
    static std::string descr(); 
};

// template specializations for code

template<> const char Desc<float>::code = 'f' ; 
template<> const char Desc<double>::code = 'f' ; 

template<> const char Desc<char>::code = 'i' ; 
template<> const char Desc<short>::code = 'i' ; 
template<> const char Desc<int>::code = 'i' ; 
template<> const char Desc<long>::code = 'i' ; 
template<> const char Desc<long long>::code = 'i' ; 

template<> const char Desc<unsigned char>::code = 'u' ; 
template<> const char Desc<unsigned short>::code = 'u' ; 
template<> const char Desc<unsigned int>::code = 'u' ; 
template<> const char Desc<unsigned long>::code = 'u' ; 
template<> const char Desc<unsigned long long>::code = 'u' ; 

template<> const char Desc<std::complex<float> >::code = 'c' ; 
template<> const char Desc<std::complex<double> >::code = 'c' ; 

template<typename T>
inline std::string Desc<T>::descr()
{
    std::stringstream ss ; 
    ss 
       << Endian::detect()
       << code 
       << sizeof(T)        
       ;
    return ss.str(); 
}


struct NPS
{
    NPS(std::vector<int>& shape_ ) : shape(shape_) {}  ; 

    static int set_shape(std::vector<int>& shape_, int ni, int nj=-1, int nk=-1, int nl=-1, int nm=-1) 
    {
        NPS sh(shape_); 
        sh.set_shape(ni,nj,nk,nl,nm); 
        return sh.size(); 
    }

    void set_shape(int ni, int nj=-1, int nk=-1, int nl=-1, int nm=-1)
    {
        if(ni > 0) shape.push_back(ni); 
        if(nj > 0) shape.push_back(nj); 
        if(nk > 0) shape.push_back(nk); 
        if(nl > 0) shape.push_back(nl); 
        if(nm > 0) shape.push_back(nm); 
    }

    static std::string desc(const std::vector<int>& shape)
    {
        std::stringstream ss ; 
        ss << "("  ; 
        for(int i=0 ; i < shape.size() ; i++) ss << shape[i] << ", " ; 
        ss << ")"  ; 
        return ss.str(); 
    } 

    static std::string json(const std::vector<int>& shape)
    {
        std::stringstream ss ; 
        ss << "["  ; 
        for(int i=0 ; i < shape.size() ; i++) 
        {
            ss << shape[i]  ; 
            if( i < shape.size() - 1 ) ss << ", " ; 
        }
        ss << "]"  ; 
        return ss.str(); 
    } 

    static int size(const std::vector<int>& shape)
    {
        int sz = 1;
        for(int i=0; i<shape.size(); ++i) sz *= shape[i] ;
        return sz ;  
    }

    std::string desc() const { return desc(shape) ; }
    std::string json() const { return json(shape) ; }
    int size() const { return size(shape) ; }

    static int ni_(const std::vector<int>& shape) { return shape.size() > 0 ? shape[0] : 1 ;  }
    static int nj_(const std::vector<int>& shape) { return shape.size() > 1 ? shape[1] : 1 ;  }
    static int nk_(const std::vector<int>& shape) { return shape.size() > 2 ? shape[2] : 1 ;  }
    static int nl_(const std::vector<int>& shape) { return shape.size() > 3 ? shape[3] : 1 ;  }
    static int nm_(const std::vector<int>& shape) { return shape.size() > 4 ? shape[4] : 1 ;  }

    int ni_() const { return ni_(shape) ; }
    int nj_() const { return nj_(shape) ; }
    int nk_() const { return nk_(shape) ; }
    int nl_() const { return nl_(shape) ; }
    int nm_() const { return nm_(shape) ; }

    int idx(int i, int j, int k, int l, int m)
    {
        int ni = ni_() ;
        int nj = nj_() ; 
        int nk = nk_() ; 
        int nl = nl_() ;
        int nm = nm_() ;

        return  i*nj*nk*nl*nm + j*nk*nl*nm + k*nl*nm + l*nm + m ;
    }


    std::vector<int>& shape ; 
};



struct U
{
    static bool EndsWith( const char* s, const char* q) ; 
    static std::string ChangeExt( const char* s, const char* x1, const char* x2) ; 
};

inline bool U::EndsWith( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ; 
}

inline std::string U::ChangeExt( const char* s, const char* x1, const char* x2)
{
    assert( EndsWith(s, x1) ); 

    std::string st = s ; 
    std::stringstream ss ; 

    ss << st.substr(0, strlen(s) - strlen(x1) ) ; 
    ss << x2 ;  
    return ss.str() ; 
}




struct NPU
{
    static const char* MAGIC ; 
    static const bool  FORTRAN_ORDER ;

    template<typename T>
    static std::string make_header(const std::vector<int>& shape );

    template<typename T>
    static std::string make_jsonhdr(const std::vector<int>& shape );

    static void parse_header(std::vector<int>& shape, std::string& descr, char& uifc, int& ebyte, const std::string& hdr );
    static int  _parse_header_length(const std::string& hdr );
    static void _parse_tuple(std::vector<int>& shape, const std::string& sh );
    static void _parse_dict(bool& little_endian, char& uifc, int& width, std::string& descr, bool& fortran_order, const char* dict); 
    static void _parse_dict(std::string& descr, bool& fortran_order, const char* dict);
    static void _parse_descr(bool& little_endian, char& uifc, int& width, const char* descr);  

    static int  _dtype_ebyte(const char* dtype);
    static char _dtype_uifc(const char* dtype);


    static std::string _make_preamble( int major=1, int minor=0 );
    static std::string _make_header(const std::vector<int>& shape, const char* descr="<f4" );
    static std::string _make_jsonhdr(const std::vector<int>& shape, const char* descr="<f4" );
    static std::string _little_endian_short_string( uint16_t dlen ) ; 
    static std::string _make_tuple(const std::vector<int>& shape, bool json );
    static std::string _make_dict(const std::vector<int>& shape, const char* descr );
    static std::string _make_json(const std::vector<int>& shape, const char* descr );
    static std::string _make_header(const std::string& dict);
    static std::string _make_jsonhdr(const std::string& json);

    static std::string xxdisplay(const std::string& hdr, int width, char non_printable );
    static std::string _check(const char* path); 
    static int         check(const char* path); 
    static bool is_readable(const char* path);
};

// inline variables requires -std=c++1z to avoid warning :  inline variables are a C++1z extension [-Wc++1z-extensions]
const char* NPU::MAGIC = "\x93NUMPY" ; 
const bool  NPU::FORTRAN_ORDER = false ; 

template<typename T>
inline std::string NPU::make_header(const std::vector<int>& shape )
{
    std::string descr = Desc<T>::descr() ;   
    return _make_header( shape, descr.c_str() ) ; 
}

template<typename T>
inline std::string NPU::make_jsonhdr(const std::vector<int>& shape )
{
    std::string descr = Desc<T>::descr() ; 
    return _make_jsonhdr( shape, descr.c_str() ) ; 
}

inline std::string NPU::xxdisplay(const std::string& hdr, int width, char non_printable)
{
    std::stringstream ss ; 
    for(int i=0 ; i < hdr.size() ; i++) 
    {   
        char c = hdr[i] ; 
        bool printable = c >= ' ' && c <= '~' ;  // https://en.wikipedia.org/wiki/ASCII
        ss << ( printable ? c : non_printable )  ;
        if((i+1) % width == 0 ) ss << "\n" ; 
   }   
   return ss.str(); 
}

inline int NPU::_parse_header_length(const std::string& hdr )
{
/*
Extract from the NPY format specification
-------------------------------------------

* https://github.com/numpy/numpy/blob/master/doc/neps/nep-0001-npy-format.rst
   
1. The first 6 bytes are a magic string: exactly "x93NUMPY".
2. The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
3. The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00. 
   Note: the version of the file format is not tied to the version of the numpy package.

4. The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.

The next HEADER_LEN bytes form the header data describing the array's format.
It is an ASCII string which contains a Python literal expression of a
dictionary. It is terminated by a newline ('n') and padded with spaces ('x20')
to make the total length of the magic string + 4 + HEADER_LEN be evenly
divisible by 16 for alignment purposes.

Example Headers
----------------

Created by commands like::

    python -c "import numpy as np ; np.save('/tmp/z0.npy', np.zeros((10,4), dtype=np.float64)) "

Older NumPy does not add padding::

    epsilon:np blyth$ xxd /tmp/z.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 6638 272c 2027 666f 7274  r': '<f8', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 302c  e, 'shape': (10,
    00000040: 2034 292c 207d 2020 2020 2020 2020 200a   4), }         .
    00000050: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000060: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000070: 0000 0000 0000 0000 0000 0000 0000 0000  ................

Newer NumPy adds a little padding to the header::

    epsilon:np blyth$ xxd /tmp/z0.npy
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6638 272c 2027 666f 7274  r': '<f8', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 302c  e, 'shape': (10,
    00000040: 2034 292c 207d 2020 2020 2020 2020 2020   4), }          
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020                  
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020                  
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000080: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    00000090: 0000 0000 0000 0000 0000 0000 0000 0000  ................

Parsing the header
-------------------

The preamble is first 8 bytes, 6 bytes for the magic then 2 bytes for the version, 
followed by 2 bytes with the header length : making 10 bytes which are always present.
The header length does not include these first 10 bytes.  The header is padded with x20
to make (hlen+10)%16 == 0 and it is terminated with a newline hex:0a dec:10  

NumPy np.save / np.load
-------------------------

* https://github.com/numpy/numpy/blob/master/numpy/lib/npyio.py
* https://github.com/numpy/numpy/blob/master/numpy/lib/format.py

*/
    std::string preamble = hdr.substr(0,8) ;  // 6 char MAGIC + 2 char version  
    std::string PREAMBLE = _make_preamble(); 
    assert( preamble.compare(PREAMBLE) == 0 );  

    char hlen_lsb = hdr[8] ;  
    char hlen_msb = hdr[9] ;  
    int hlen = hlen_msb << 8 | hlen_lsb ; 
    assert( (hlen+10) % 16 == 0 ) ;  
    assert( hlen+10 == hdr.size() ) ; 

#ifdef NPU_DEBUG
    std::cout 
        << " _parse_header_length  "  << std::endl 
        << " hdr               " << std::endl << xxdisplay(hdr, 16, '.' ) << std::endl  
        << " preamble          " << preamble << std::endl 
        << " hlen_lsb(hex)     " << std::hex << int(hlen_lsb) << std::endl 
        << " hlen_msb(hex)     " << std::hex << int(hlen_msb) << std::endl 
        << " hlen(hex)         " << std::hex << hlen << std::endl 
        << " hlen_lsb(dec)     " << std::dec << int(hlen_lsb) << std::endl 
        << " hlen_msb(dec)     " << std::dec << int(hlen_msb) << std::endl 
        << " hlen(dec)         " << std::dec << hlen << std::endl 
        << " hlen+10(dec)      " << std::dec << hlen+10 << std::endl 
        << " (hlen+10)%16(dec) " << (hlen+10)%16 << std::endl 
        << " hdr.size() (dec)  " << std::dec << hdr.size() << std::endl 
        << " preamble.size()   " << std::dec << preamble.size() << std::endl 
        << std::endl 
        ; 

#endif
    return hlen ; 
}


inline void NPU::parse_header(std::vector<int>& shape, std::string& descr, char& uifc, int& ebyte, const std::string& hdr )
{
    int hlen = _parse_header_length( hdr ) ; 

    std::string dict = hdr.substr(10,10+hlen) ; 

    char last = dict[dict.size()-1] ; 
    bool ends_with_newline = last == '\n' ;   
    assert(ends_with_newline) ; 
    dict[dict.size()-1] = '\0' ; 

    std::string::size_type p0 = dict.find("(") + 1; 
    std::string::size_type p1 = dict.find(")"); 
    assert( p0 != std::string::npos ); 
    assert( p1 != std::string::npos ); 

    std::string sh = dict.substr( p0, p1 - p0 ) ;  

    _parse_tuple( shape, sh ); 


    bool little_endian ; 
    bool fortran_order ; 
  
    _parse_dict(little_endian, uifc, ebyte, descr, fortran_order, dict.c_str());


    assert( fortran_order == FORTRAN_ORDER ); 
    assert( little_endian == true ); 

#ifdef NPU_DEBUG
    std::cout 
        << " parse_header  "  << std::endl 
        << " hdr               " << std::endl << xxdisplay(hdr, 16, '.' ) << std::endl  
        << " hlen(hex)         " << std::hex << hlen << std::endl 
        << " hlen(dec)         " << std::dec << hlen << std::endl 
        << " hlen+10(dec)      " << std::dec << hlen+10 << std::endl 
        << " (hlen+10)%16(dec) " << (hlen+10)%16 << std::endl 
        << " dict [" << xxdisplay(dict,200,'.') << "]"<< std::endl 
        << " p0( " << p0 << std::endl
        << " p1) " << p1 << std::endl
        << " shape " << sh << std::endl
        << " last(dec)         " << std::dec << int(last) << std::endl 
        << " newline(dec)      " << std::dec << int('\n') << std::endl 
        << " hdr.size() (dec)  " << std::dec << hdr.size() << std::endl 
        << " dict.size() (dec) " << std::dec << dict.size() << std::endl 
        << " descr " << descr
        << " uifc " << uifc
        << " ebyte " << ebyte 
        << std::endl 
        ; 

#endif

}

inline void NPU::_parse_tuple(std::vector<int>& shape, const std::string& sh )
{
    std::istringstream f(sh);
    std::string s;

    char delim = ',' ; 
    const char* trim = " " ;  

    int ival(0) ; 

    while (getline(f, s, delim)) 
    {
       s.erase(0, s.find_first_not_of(trim));  // left trim
       s.erase(s.find_last_not_of(trim) + 1);   // right trim 
       if( s.size() == 0 ) continue ; 

       std::istringstream ic(s) ;
       ic >> ival ; 

       shape.push_back(ival) ; 
 
#ifdef NPU_DEBUG
       std::cout << "[" << s << "] -> " << ival << std::endl ;
#endif

    }

#ifdef NPU_DEBUG
    std::cout << " parse_tuple " 
              << " sh  [" << sh << "]" 
              << " shape " << shape.size()
              << std::endl
              ;

#endif
}


inline void NPU::_parse_dict(bool& little_endian, char& uifc, int& ebyte, std::string& descr, bool& fortran_order, const char* dict)  // static 
{
    _parse_dict(descr, fortran_order, dict); 
    _parse_descr(little_endian, uifc, ebyte, descr.c_str() ); 
}


/**
NPU::_parse_dict
------------------

::

    const char* dict = R"({'descr': '<f4', 'fortran_order': False, 'shape': (10, 4), })" ; 
    //       nq:           1     2  3   4  5             6         7     8
    //     elem:                 0      1                2       3       4 

**/

inline void NPU::_parse_dict(std::string& descr, bool& fortran_order, const char* dict) // static
{
    char q = '\'' ;  
    char x = '\0' ;   // "wildcard" extra delim 

    std::vector<std::string> elem ;  
    std::stringstream ss ; 
    unsigned nq = 0 ; 
    for(int i=0 ; i < strlen(dict) ; i++)
    {
        if(dict[i] == q || dict[i] == x) 
        {
            nq += 1 ;  
            if(nq == 6 ) x = ' ' ; 
            if(nq == 7 ) x = ',' ; 
            if(nq == 8 ) x = '\0' ; 

            if( nq % 2 == 0 )  
            {
                elem.push_back(ss.str());  
                ss.str("");
            }
        } 
        else
        {
            if(nq % 2 == 1 ) ss << dict[i] ; 
        }
    }

    assert( elem[0].compare("descr") == 0 );  
    assert( elem[2].compare("fortran_order") == 0 );  
    assert( elem[3].compare("False") == 0 || elem[3].compare("True") == 0);  
    assert( elem[4].compare("shape") == 0 );  

    descr = elem[1];
    assert( descr.length() == 3 ); 
 
    fortran_order = elem[3].compare("False") == 0 ? false : true ; 
}

inline void NPU::_parse_descr(bool& little_endian, char& uifc, int& ebyte, const char* descr)  // static
{
    assert( strlen(descr) == 3 ); 

    char c_endian = descr[0] ; 
    char c_uifc = descr[1] ; 
    char c_ebyte = descr[2] ; 

    bool expect_endian = c_endian == '<' || c_endian == '>' || c_endian == '|' ; 
    if(!expect_endian)
    {
        std::cerr 
            << "unexpected endian "
            << " c_endian " << c_endian 
            << " descr [" << descr << "]"  
            << std::endl
             ; 
    }
    assert( expect_endian ); 
    little_endian = c_endian == '<' || c_endian == '|' ;

    assert( c_uifc == 'u' || c_uifc == 'i' || c_uifc == 'f' || c_uifc == 'c' ); 
    uifc = c_uifc ; 

    ebyte = c_ebyte - '0' ; 
    assert( ebyte == 1 || ebyte == 2 || ebyte == 4 || ebyte == 8 ); 
}

inline int NPU::_dtype_ebyte(const char* dtype)  // static 
{
    unsigned len = strlen(dtype) ; 
    assert( len == 2 || len == 3 ); 

    char c_ebyte = dtype[len-1] ;  
    int ebyte = c_ebyte - '0' ; 
    
    assert( ebyte == 1 || ebyte == 2 || ebyte == 4 || ebyte == 8 ); 
    return ebyte ; 
} 
inline char NPU::_dtype_uifc(const char* dtype) // static
{
    unsigned len = strlen(dtype) ; 
    assert( len == 2 || len == 3 ); 
    char c_uifc = dtype[len-2] ; 
    assert( c_uifc == 'u' || c_uifc == 'i' || c_uifc == 'f' );  // dont bother with 'c' complex  
    return c_uifc ; 
}


 
inline bool NPU::is_readable(const char* path)  // static 
{
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    bool readable = !fp.fail(); 
    fp.close(); 
    return readable ; 
}


inline std::string NPU::_check(const char* path) 
{
    char* py = getenv("PYTHON"); 
    std::stringstream ss ; 
    ss << ( py ? py : "python" )
       << " -c \"import numpy as np ; print(np.load('" 
       << path 
       << "')) \" && xxd " 
       << path 
       ; 
    return ss.str(); 
}

inline int NPU::check(const char* path)
{
    std::string cmd = _check(path); 
    return system(cmd.c_str()); 
}



inline std::string NPU::_make_header(const std::vector<int>& shape, const char* descr )
{
    std::string dict = _make_dict( shape, descr ); 
    std::string header = _make_header( dict ); 
    return header ; 
}

inline std::string NPU::_make_jsonhdr(const std::vector<int>& shape, const char* descr )
{
    std::string json = _make_json( shape, descr ); 
    return json ; 
}



inline std::string NPU::_make_dict(const std::vector<int>& shape, const char* descr )
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

inline std::string NPU::_make_json(const std::vector<int>& shape, const char* descr )
{
    std::stringstream ss ; 
    ss << "{" ; 
    ss << "\"descr\": \"" << descr << "\", " ; 
    ss << "\"fortran_order\": " << ( FORTRAN_ORDER ? "true" : "false" ) << ", " ; 
    ss << "\"shape\": " ; 
    bool json = true ; 
    ss << _make_tuple( shape, json) ; 
    ss << "}" ;  
    return ss.str(); 
} 




inline std::string NPU::_make_tuple( const std::vector<int>& shape, bool json )
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



inline std::string NPU::_little_endian_short_string( uint16_t dlen )
{
    // https://github.com/numpy/numpy/blob/master/doc/neps/nep-0001-npy-format.rst
    // The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN

    u16c2_t len ; 
    len.u16 = dlen ; 

    char e = Endian::detect() ; 
    std::string hlen(2, ' ') ;
    hlen[0] = e == Endian::LITTLE ? len.c[0] : len.c[1] ;  
    hlen[1] = e == Endian::LITTLE ? len.c[1] : len.c[0] ; 

#ifdef NPU_DEBUG
    std::cout << " dlen " << dlen << std::endl ; 
    std::cout << " len.c[0] " << len.c[0] << std::endl ; 
    std::cout << " len.c[1] " << len.c[1] << std::endl ; 
    std::cout << ( e == Endian::LITTLE ? "little_endian" : "big_endian" ) << std::endl ; 
#endif

    return hlen ; 
}


inline std::string NPU::_make_preamble( int major, int minor )
{
    std::string preamble(MAGIC) ; 
    preamble.push_back((char)major); 
    preamble.push_back((char)minor); 
    return preamble ; 
}

inline std::string NPU::_make_header(const std::string& dict)
{
    uint16_t dlen = dict.size() ;
    uint16_t padding = 16 - ((10 + dlen ) % 16 ) - 1 ;
    padding += 3*16 ; // adhoc extra padding for bit-perfect matching to NumPy (for test array anyhow)
    uint16_t hlen = dlen + padding + 1 ; 

#ifdef NPU_DEBUG
    std::cout 
        << " dlen " << dlen 
        << " padding " << padding
        << " hlen " << hlen 
        << std::endl 
        ; 
#endif

    assert( (hlen + 10) % 16 == 0 );  
    std::stringstream ss ; 
    ss << _make_preamble() ;  
    ss << _little_endian_short_string( hlen ) ; 
    ss << dict ; 
 
    for(int i=0 ; i < padding ; i++ ) ss << " " ; 
    ss << "\n" ;  

    return ss.str(); 
}



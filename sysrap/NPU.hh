#pragma once

/**
NPU.hh : Utilities used from NP.hh
====================================

This is developed in https://github.com/simoncblyth/np/
but given the header-only nature is often just incorporated into 
other projects together with NP.hh

**/


#include <csignal>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>
#include <complex>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <locale>


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


/**
desc : type codes and sizes used by descr_
---------------------------------------------
**/

template<typename T>
struct desc 
{
    static constexpr char code = '?' ; 
    static constexpr unsigned size = 0 ; 
};

template<> struct desc<float>  { static constexpr char code = 'f' ; static constexpr unsigned size = sizeof(float)  ; };
template<> struct desc<double> { static constexpr char code = 'f' ; static constexpr unsigned size = sizeof(double) ; };

template<> struct desc<char> {   static constexpr char code = 'i' ; static constexpr unsigned size = sizeof(char)   ; };
template<> struct desc<short> {  static constexpr char code = 'i' ; static constexpr unsigned size = sizeof(short)  ; };
template<> struct desc<int> {    static constexpr char code = 'i' ; static constexpr unsigned size = sizeof(int)    ; };
template<> struct desc<long> {   static constexpr char code = 'i' ; static constexpr unsigned size = sizeof(long)   ; };
template<> struct desc<long long> {  static constexpr char code = 'i' ; static constexpr unsigned size = sizeof(long long)   ;  };

template<> struct desc<unsigned char> {   static constexpr char code = 'u' ; static constexpr unsigned size = sizeof(unsigned char) ;  };
template<> struct desc<unsigned short> {  static constexpr char code = 'u' ; static constexpr unsigned size = sizeof(unsigned short) ;   };
template<> struct desc<unsigned int> {    static constexpr char code = 'u' ; static constexpr unsigned size = sizeof(unsigned int) ;  };
template<> struct desc<unsigned long> {   static constexpr char code = 'u' ; static constexpr unsigned size = sizeof(unsigned long) ;   };
template<> struct desc<unsigned long long> {  static constexpr char code = 'u' ; static constexpr unsigned size = sizeof(unsigned long long) ;  };

template<> struct desc<std::complex<float> > {   static constexpr char code = 'c' ; static constexpr unsigned size = sizeof(std::complex<float>)  ; } ; 
template<> struct desc<std::complex<double> > {  static constexpr char code = 'c' ; static constexpr unsigned size = sizeof(std::complex<double>) ; } ; 

struct endian
{
    static char constexpr LITTLE = '<' ; 
    static char constexpr BIG = '>' ; 
    static char detect() { unsigned one = 1u ; return (*(char *)&one == 1) ? LITTLE : BIG ; } ;
};

template<typename T>
struct descr_
{
    static std::string dtype()
    {
        std::stringstream ss ; 
        ss << endian::detect() << desc<T>::code << desc<T>::size ;
        return ss.str(); 
    }
};

template struct descr_<float> ;  
template struct descr_<double> ;
  
template struct descr_<char> ;  
template struct descr_<short> ;  
template struct descr_<int> ;  
template struct descr_<long> ;  
template struct descr_<long long> ;  
 
template struct descr_<unsigned char> ;  
template struct descr_<unsigned short> ;  
template struct descr_<unsigned int> ;  
template struct descr_<unsigned long> ;  
template struct descr_<unsigned long long> ;  

template struct descr_<std::complex<float> > ;  
template struct descr_<std::complex<double> > ;  






/**
net_hdr
---------

Packing and unpacking of simple network header
composed of a small number of 32bit unsigned ints 
expressed in big endian "network order". 

**/

#include <arpa/inet.h>    // htonl


struct net_hdr
{
    static constexpr unsigned LENGTH = 4*4 ; 

    union uc4_t {
        uint32_t          u    ;   
        char              c[4] ; 
    };  
    static std::string pack(const std::vector<unsigned> items);
    static void unpack(     const std::string& hdr         , std::vector<unsigned>& items );
    static void unpack( char* data, unsigned num_bytes , std::vector<unsigned>& items );

    static unsigned unpack( const std::string& hdr, unsigned index ); 
};


inline std::string net_hdr::pack(const std::vector<unsigned> items) // static 
{
    unsigned ni = items.size(); 

    assert( ni == 4 ); 
    assert( sizeof(unsigned) == 4); 
    assert( ni*sizeof(unsigned) == LENGTH ); 

    uc4_t uc4 ; 
    std::string hdr(LENGTH, '\0' );  
    for(unsigned i=0 ; i < ni ; i++)
    {   
        uc4.u = htonl(items[i]) ;   // to big endian or "network order"
        memcpy( (void*)(hdr.data() + i*sizeof(unsigned)), &(uc4.c[0]), 4 );  
    }   
    return hdr ; 
}

inline void net_hdr::unpack( const std::string& hdr, std::vector<unsigned>& items ) // static
{
    unpack((char*)hdr.data(), hdr.length(), items );
}

inline unsigned net_hdr::unpack( const std::string& hdr, unsigned index ) // static
{
    std::vector<unsigned> items ; 
    unpack(hdr, items); 
    return index < items.size() ? items[index] : 0 ; 
} 

inline void net_hdr::unpack( char* data, unsigned num_bytes, std::vector<unsigned>& items ) // static
{
    assert( 4 == sizeof(unsigned)); 
    unsigned ni = num_bytes/sizeof(unsigned); 

    items.clear(); 
    items.resize(ni); 

    uc4_t uc4 ; 
    for(unsigned i=0 ; i < ni ; i++)
    {   
        memcpy( &(uc4.c[0]), data + i*4, 4 );  
        items[i] = ntohl(uc4.u) ;   // from big endian to endian-ness of host  
    }   
}



struct NPS
{
    NPS(std::vector<int>& shape_ ) : shape(shape_) {}  ; 

    static int set_shape(std::vector<int>& shape_, int ni, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ) 
    {
        NPS sh(shape_); 
        sh.set_shape(ni,nj,nk,nl,nm,no); 
        return sh.size(); 
    }

    static int copy_shape(std::vector<int>& dst, const std::vector<int>& src) 
    {
        for(unsigned i=0 ; i < src.size() ; i++) dst.push_back(src[i]); 
        return size(dst); 
    }

    static int copy_shape(std::vector<int>& dst, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1) 
    {
        if(ni >= 0) dst.push_back(ni);   // experimental allow zero items
        if(nj > 0) dst.push_back(nj); 
        if(nk > 0) dst.push_back(nk); 
        if(nl > 0) dst.push_back(nl); 
        if(nm > 0) dst.push_back(nm); 
        if(no > 0) dst.push_back(no); 
        return size(dst); 
    }

    void set_shape(int ni, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1)
    {
        if(ni >= 0) shape.push_back(ni);   // experimental allow zero items
        if(nj > 0) shape.push_back(nj); 
        if(nk > 0) shape.push_back(nk); 
        if(nl > 0) shape.push_back(nl); 
        if(nm > 0) shape.push_back(nm); 
        if(no > 0) shape.push_back(no); 
    }
    void set_shape(const std::vector<int>& other)
    {
        copy_shape(shape, other); 
    }

    static int change_shape(std::vector<int>& shp, int ni_, int nj_=-1, int nk_=-1, int nl_=-1, int nm_=-1, int no_=-1)
    {
        unsigned nv0 = size(shp); 
        unsigned nv1 = std::max(1,ni_)*std::max(1,nj_)*std::max(1,nk_)*std::max(1,nl_)*std::max(1,nm_)*std::max(1,no_) ; 

        if( nv0 != nv1 )  // try to devine a missing -1 entry 
        {
            if(      ni_ < 0 ) ni_ = nv0/nv1 ; 
            else if( nj_ < 0 ) nj_ = nv0/nv1 ; 
            else if( nk_ < 0 ) nk_ = nv0/nv1 ; 
            else if( nl_ < 0 ) nl_ = nv0/nv1 ; 
            else if( nm_ < 0 ) nm_ = nv0/nv1 ; 
            else if( no_ < 0 ) no_ = nv0/nv1 ; 

            unsigned nv2 = std::max(1,ni_)*std::max(1,nj_)*std::max(1,nk_)*std::max(1,nl_)*std::max(1,nm_)*std::max(1,no_) ; 
            bool expect = nv0 % nv1 == 0 && nv2 == nv0 ; 

            if(!expect) std::cout 
                << " NPS::change_shape INVALID SHAPE CHANGE : SIZE MUST STAY CONSTANT : ONLY ONE -1 ENTRY CAN BE AUTO-FILLED  "
                << std::endl 
                << " nv0 " << nv0
                << " nv1 " << nv1
                << " nv2 " << nv2
                << " ni_ " << ni_
                << " nj_ " << nj_
                << " nk_ " << nk_
                << " nl_ " << nl_
                << " nm_ " << nm_
                << " no_ " << no_
                << std::endl
                ;

            assert(expect); 
        }
         
        shp.clear(); 
        return copy_shape(shp, ni_, nj_, nk_, nl_, nm_, no_ ); 
    }

    static int product(const std::vector<int>& src )
    {
        int nd = src.size(); 
        int prod = 1 ; 
        for(int i=0 ; i < nd ; i++) prod *= src[i] ; 
        return prod ; 
    }
    static void reshape(std::vector<int>& dst, const std::vector<int>& src )
    {
        assert( product(dst) == product(src) ); 
        dst = src ; 
    }

    template<int P>
    static void size_2D( int& width, int& height, const std::vector<int>& sh ) 
    {
        int nd = sh.size() ; 
        assert( nd > 1 && sh[nd-1] == P );  
        width = sh[nd-2] ; 
        height = 1 ; 
        for(int i=0 ; i < nd-2 ; i++) height *= sh[i] ; 
    }

    static std::string desc(const std::vector<int>& shape)
    {
        std::stringstream ss ; 
        ss << "("  ; 
        for(unsigned i=0 ; i < shape.size() ; i++) ss << shape[i] << ", " ; 
        ss << ")"  ; 
        return ss.str(); 
    } 

    static std::string json(const std::vector<int>& shape)
    {
        std::stringstream ss ; 
        ss << "["  ; 
        for(unsigned i=0 ; i < shape.size() ; i++) 
        {
            ss << shape[i]  ; 
            if( i < shape.size() - 1 ) ss << ", " ; 
        }
        ss << "]"  ; 
        return ss.str(); 
    } 

    static int size(const std::vector<int>& shape)
    {
        int ndim = int(shape.size()); 
        int sz = 1;
        for(int i=0; i<ndim; ++i) sz *= shape[i] ;
        return ndim == 0 ? 0 : sz ;  
    }

    static int itemsize(const std::vector<int>& shape)
    {
        int sz = 1;
        for(unsigned i=1; i<shape.size(); ++i) sz *= shape[i] ;
        return sz ;  
    }

    static int itemsize_(const std::vector<int>& shape, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1 )
    {
        // assert only one transition from valid indices to skipped indices 
        if( i == -1 )                                                      assert( j == -1 && k == -1 &&  l == -1 && m == -1 && o == -1 ) ;  
        if( i > -1 && j == -1 )                                            assert(            k == -1 &&  l == -1 && m == -1 && o == -1 ) ;  
        if( i > -1 && j > -1 && k == -1 )                                  assert(                        l == -1 && m == -1 && o == -1 ) ;  
        if( i > -1 && j > -1 && k >  -1 && l == -1 )                       assert(                                   m == -1 && o == -1 ) ;  
        if( i > -1 && j > -1 && k >  -1 && l >  -1 && m == -1 )            assert(                                              o == -1 ) ;  
        if( i > -1 && j > -1 && k >  -1 && l >  -1 && m >  -1 && o == -1 ) assert(                                              true    ) ;  

        unsigned dim0 = 0 ; 
        if( i == -1 )                                                      dim0 = 0 ; 
        if( i > -1 && j == -1 )                                            dim0 = 1 ; 
        if( i > -1 && j > -1 && k == -1 )                                  dim0 = 2 ; 
        if( i > -1 && j > -1 && k >  -1 && l == -1 )                       dim0 = 3 ;  
        if( i > -1 && j > -1 && k >  -1 && l >  -1 && m == -1 )            dim0 = 4 ; 
        if( i > -1 && j > -1 && k >  -1 && l >  -1 && m >  -1 && o == -1 ) dim0 = 5 ; 
        if( i > -1 && j > -1 && k >  -1 && l >  -1 && m >  -1 && o >  -1 ) dim0 = 6 ; 

        int sz = 1;
        if( dim0 < shape.size() )
        {
            for(unsigned d=dim0; d<shape.size(); ++d) sz *= shape[d] ;
        }
#ifdef DEBUG_NPU
        std::cout 
            << "NPS::itemsize_"
            << "(" << std::setw(3) << i 
            << " " << std::setw(3) << j
            << " " << std::setw(3) << k
            << " " << std::setw(3) << l
            << " " << std::setw(3) << m
            << " " << std::setw(3) << o
            << ")"
            << " " << sz 
            << std::endl 
            ; 
#endif
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
    static int no_(const std::vector<int>& shape) { return shape.size() > 5 ? shape[5] : 1 ;  }

    int ni_() const { return ni_(shape) ; }
    int nj_() const { return nj_(shape) ; }
    int nk_() const { return nk_(shape) ; }
    int nl_() const { return nl_(shape) ; }
    int nm_() const { return nm_(shape) ; }
    int no_() const { return no_(shape) ; }

    int idx(int i, int j, int k, int l, int m, int o)
    {
        //int ni = ni_() ;
        int nj = nj_() ; 
        int nk = nk_() ; 
        int nl = nl_() ;
        int nm = nm_() ;
        int no = no_() ;

        return  i*nj*nk*nl*nm*no + j*nk*nl*nm*no + k*nl*nm*no + l*nm*no + m*no + o ;
    }


    std::vector<int>& shape ; 
};



struct U
{
    static constexpr const bool VERBOSE = false ; 
    static constexpr const bool RAISE = true ; 

    enum { ERROR_PATH=-1, DIR_PATH=1 , FILE_PATH=2, OTHER_PATH=3 } ;   

    static void sizeof_check(); 

    static bool EndsWith( const char* s, const char* q) ; 
    static std::string ChangeExt( const char* s, const char* x1, const char* x2) ; 
    static std::string DirName( const char* path ); 

    static std::string BaseName( const char* path ); 
    static const char* BaseName_( const char* path ); 

    static std::string BaseName_NoSepAsis( const char* path ); 
    static const char* BaseName_NoSepAsis_( const char* path ); 


    static std::string FormSiblingPath( const char* sibname , const char* dirpath ); 
    static std::string FormExecutableSiblingPath( const char* argv0 , const char* dirpath ); 
    static bool        IsExecutableSiblingPath(   const char* argv0,  const char* dirpath ); 
    static int SetEnvDefaultExecutableSiblingPath(const char* ekey, char* argv0, const char* dirpath ); 

    static int setenvvar( const char* ekey, const char* value, bool overwrite=true, char special_empty_token='\0' );

    static std::string FormName_( int idx, int wid=3 ); 
    static const char* FormName( int idx, int wid=3 ); 

    static std::string FormName_( const char* prefix, int idx, const char* ext, int wid=-1 ); 
    static const char* FormName(  const char* prefix, int idx, const char* ext, int wid=-1 ); 

    static std::string FormName_( const char* prefix, const char* body, const char* ext ); 
    static const char* FormName( const char* prefix, const char* body, const char* ext ); 

    static bool isdigit_(char c ); 
    static bool isalnum_(char c ); 
    static bool isupper_(char c ); 
    static bool islower_(char c ); 


    static void Summarize( std::vector<std::string>& smry_labels, const std::vector<std::string>* labels, int wid ); 
    static const char* Summarize( const char* label, int wid ); 
    static std::string Summarize_( const char* label, int wid ); 


    static void LineVector( std::vector<std::string>& lines, const char* LINES ); 

    static void LiteralTrim( std::string& line ); 
    static void Literal(    std::vector<std::string>& lines, const char* LINES ); 
    static void LiteralAnno( std::vector<std::string>& field, std::vector<std::string>& anno, const char* LINES, const char* delim="#" );

    static std::string Space(int wid); 



    static std::string form_name(const char* stem, const char* ext); 
    static std::string form_path(const char* dir, const char* name);   
    static std::string form_path(const char* dir, const char* reldir, const char* name);   



    static constexpr const char* DEFAULT_PATH_ARG_0 = "/tmp" ; 
    template<typename ... Args>
    static std::string Path_( Args ... args_  ); 

    template<typename ... Args>
    static const char* Path( Args ... args ); 


    template<typename T>
    static inline void MakeVec(std::vector<T>& vec, const char* line, char delim=','); 

    template<typename T>
    static std::vector<T>* MakeVec(const char* line, char delim=','); 

    static void Zip(
        std::vector<std::string>& kvs, 
        const std::vector<std::string>& keys, 
        const std::vector<std::string>& vals, 
        char delim=':'); 


    template<typename T>
    static long LoadVec(std::vector<T>& vec, const char* path_); 

    template<typename T>
    static int Category(const std::vector<T>& cats, const T& val ); 

    static bool StartsWith( const char* s, const char* q) ; 
    static bool Contains(   const char* s, const char* q) ; 

    template<typename T> static unsigned NumSteps( T x0, T x1, T dx ); 

    template<typename T> static T To( const char* a ); 
    template<typename T> static bool ConvertsTo( const char* a ); 

    static char* PWD(); 

    template<typename T>
    static std::vector<T>* GetEnvVec(const char* ekey, const char* fallback, char delim=','); 
    static int         GetEnvInt( const char* envkey, int fallback );  
    static const char* GetEnv(    const char* envkey, const char* fallback); 

    template<typename T>
    static T           GetE(const char* ekey, T fallback);  

    static int MakeDirs( const char* dirpath, int mode=0 ); 
    static int MakeDirsForFile( const char* filepath, int mode=0); 

    static int PathType( const char* path );  // directory:1 file:2 
    static int PathType( const char* base, const char* name );  // directory:1 file:2 

    static void DirList(std::vector<std::string>& names, const char* path, const char* ext=nullptr, bool exclude=false ); 
    static void Trim(std::vector<std::string>& names, const char* ext); 
    static void Split(const char* str, char delim,   std::vector<std::string>& elem); 
    static int FindIndex(const std::vector<std::string>& names, const char* name);

    static std::string Desc(const std::vector<std::string>& names); 

    static const char* Resolve0(const char* spec, const char* relp=nullptr ); 
    // $TOK/remainder/path.npy 

    static const char* Resolve( const char* spec, const char* rel1=nullptr, const char* rel2=nullptr ); 
    // $TOK/remainder/$ANOTHER/path.npy 

    static void        WriteString( const char* dir, const char* reldir, const char* name, const char* str ); 
    static void        WriteString( const char* dir, const char* name, const char* str ); 
    static void        WriteString( const char* path, const char* str ); 

    static const char* ReadString( const char* dir, const char* reldir, const char* name);
    static const char* ReadString( const char* dir, const char* name);
    static const char* ReadString( const char* path );

    static const char* ReadString2( const char* path );

    static uint64_t Now(); 
    static bool LooksLikeStampInt(   const char* str); 
    template<typename T>
    static bool LooksLikeTimestamp( T value ); 

    static bool LooksLikeProfileTriplet(const char* str); 

    static std::string Format(uint64_t t=0, const char* fmt="%FT%T."); 
    static std::string FormatInt(int64_t t, int wid ); 

    static char* LastDigit(const char* str); 
    static char* FirstDigit(const char* str); 
    static char* FirstToLastDigit(const char* str); 

};



inline void U::sizeof_check() // static 
{
    assert( sizeof(float) == 4  );  
    assert( sizeof(double) == 8  );  

    assert( sizeof(char) == 1 );  
    assert( sizeof(short) == 2 );
    assert( sizeof(int)   == 4 );
    assert( sizeof(long)  == 8 );
    assert( sizeof(long long)  == 8 );
}


template<typename T>
inline void U::MakeVec(std::vector<T>& vec, const char* line, char delim)
{
    if(line == nullptr) return ; 
    std::stringstream ss; 
    ss.str(line);
    std::string s;
    while (std::getline(ss, s, delim)) 
    {   
        std::istringstream iss(s);
        T t ; 
        iss >> t ; 
        vec.push_back(t) ; 
    }   
}

inline void U::Zip(
    std::vector<std::string>& kvs, 
    const std::vector<std::string>& keys, 
    const std::vector<std::string>& vals, 
    char delim )
{
    assert( keys.size() == vals.size() ); 
    kvs.clear(); 

    char s_delim[2] ; 
    s_delim[0] = delim ; 
    s_delim[1] = '\0' ;

    for(int i=0 ; i < int(keys.size()) ; i++)
    {
        std::string kv = U::FormName_( keys[i].c_str(), s_delim, vals[i].c_str() ); 
        kvs.push_back(kv);  
    }
}



/**
U::MakeVec
------------

If no elements are parsed from the line, nullptr is returned. 

**/

template<typename T>
inline std::vector<T>* U::MakeVec(const char* line, char delim)
{
    if(line == nullptr) return nullptr ; 
    std::vector<T> vec ; 
    MakeVec(vec, line, delim) ; 
    return vec.size() == 0 ? nullptr : new std::vector<T>(vec) ; 
}

/**
U::LoadVec
------------

Load bytes from binary file into vector that is sized accordingly. 
The type is expected to be "char" or "unsigned char" 

**/
template<typename T>
inline long U::LoadVec(std::vector<T>& vec, const char* path_)
{
    assert( sizeof(T) == 1 ) ; 

    const char* path = U::Resolve(path_); 
    FILE *fp = fopen(path,"rb");

    fseek(fp, 0L, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    vec.resize(file_size); 

    long bytes_read = fread(vec.data(), sizeof(T), file_size, fp );
    fclose(fp);
    assert( file_size == bytes_read ); 

    return bytes_read ; 
}




template<typename T>
inline int U::Category(const std::vector<T>& cats, const T& val )
{
    int cat = std::distance( cats.begin(), std::find(cats.begin(), cats.end(), val ) );
    if( cat == int(cats.size()) ) cat = -1 ;  
    return cat ; 
}




inline bool U::StartsWith( const char* s, const char* q) // static
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}
inline bool U::Contains( const char* s, const char* q) // static
{
    return s && q && strlen(q) <= strlen(s) && strstr(s, q) != nullptr ;
}



template <typename T> unsigned U::NumSteps( T x0, T x1, T dx )
{
    assert( x1 > x0 ); 
    assert( dx > T(0.) ) ; 

    unsigned ns = 0 ; 
    for(T x=x0 ; x <= x1 ; x+=dx ) ns+=1 ; 
    return ns ; 
}


template <typename T> inline T U::To( const char* a )   // static 
{   
    std::string s(a);
    std::istringstream iss(s);
    T v ;   
    iss >> v ; 
    return v ; 
}

// specialization for std::string as the above truncates at the first blank in the string, see tests/NP_set_meta_get_meta_test.cc  
template<> inline std::string U::To(const char* a ) 
{
    std::string s(a); 
    return s ; 
}





template <typename T> inline bool U::ConvertsTo( const char* a )   // static 
{   
    if( a == nullptr ) return false ; 
    if( strlen(a) == 0) return false ; 
    std::string s(a);
    std::istringstream iss(s);
    T v ;   
    iss >> v ; 
    return iss.fail() == false ; 
}


inline char* U::PWD() // static
{
    return getenv("PWD"); 
}

template<typename T>
inline std::vector<T>* U::GetEnvVec(const char* ekey, const char* fallback, char delim)
{
    char* line = getenv(ekey);
    return MakeVec<T>( line ? line : fallback, delim  ) ; 
}


inline int U::GetEnvInt(const char* envkey, int fallback)
{
    char* val = getenv(envkey);
    int ival = val ? std::atoi(val) : fallback ;
    return ival ; 
}

inline const char* U::GetEnv(const char* envkey, const char* fallback)
{
    const char* evalue = getenv(envkey);
    return evalue ? evalue : fallback ; 
} 


template<typename T>
inline T U::GetE(const char* ekey, T fallback)
{
    char* v = getenv(ekey);
    if(v == nullptr) return fallback ;

    std::string s(v);
    std::istringstream iss(s);
    T t ;
    iss >> t ;
    return t ;
}

template int      U::GetE(const char*, int );
template unsigned U::GetE(const char*, unsigned );
template float    U::GetE(const char*, float );
template double   U::GetE(const char*, double );
template char     U::GetE(const char*, char );
template unsigned char U::GetE(const char*, unsigned char );







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

inline std::string U::DirName( const char* path )
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/") ; 
    return pos == std::string::npos ? "" : p.substr(0, pos); 
}

inline std::string U::BaseName( const char* path )
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/") ; 
    return pos == std::string::npos ? "" : p.substr(pos+1); 
}
inline const char* U::BaseName_( const char* path )
{
    std::string name = BaseName(path); 
    return strdup(name.c_str()); 
}


/**
U::BaseName_NoSepAsis
-----------------------

Returns the basename of a path, when there is no separator 
returns the path asis. For example::

   U::BaseName_NoSepAsis("/some/directory/path/name.txt") -> "name.txt"
   U::BaseName("/some/directory/path/name.txt") -> name.txt

   U::BaseName_NoSepAsis("name.txt") -> "name.txt"
   U::BaseName("name.txt") -> ""

**/

inline std::string U::BaseName_NoSepAsis( const char* path )
{
    std::string p = path ; 
    std::size_t pos = p.find_last_of("/") ; 
    return pos == std::string::npos ? p : p.substr(pos+1); 
}
inline const char* U::BaseName_NoSepAsis_( const char* path )
{
    std::string name = BaseName_NoSepAsis(path); 
    return strdup(name.c_str()); 
}





/**
U::FormSiblingPath
---------------------

For example::

   sibname : sreport
   dirpath : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0
   returns : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL0_sreport   (now)
   returns : /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/sreport   (formerly)

**/

inline std::string U::FormSiblingPath( const char* sibname , const char* dirpath )
{
    std::stringstream ss ; 
    /*
    std::string container = DirName(dirpath) ; 
    ss << container << "/" << sibname ; 
    */    
    ss << dirpath << "_" << sibname ; 

    std::string str = ss.str(); 
    return str ; 
}


inline std::string U::FormExecutableSiblingPath( const char* argv0 , const char* dirpath )
{
    const char* exename = BaseName_NoSepAsis_(argv0) ; 
    std::string sibpath = FormSiblingPath( exename, dirpath ); 

    if(VERBOSE) std::cout 
        << "[U::FormExecutableSiblingPath"
        << std::endl 
        << " argv0 " << ( argv0 ? argv0 : "-" ) 
        << std::endl 
        << " dirpath " << ( dirpath ? dirpath : "-" )
        << std::endl 
        << " exename " << ( exename ? exename : "-" )
        << std::endl 
        << " sibpath " << sibpath 
        << std::endl 
        << "]U::FormExecutableSiblingPath"
        << std::endl 
        ;

    return sibpath ; 
}

/**
U::IsExecutableSiblingPath
----------------------------

For argv0 of sreport OR /some/path/to/sreport dirpath the
exepected results are:

* /some/director/tree/leading/to/ALL0_sreport   YES
* /some/director/tree/leading/to/sreport        NO

* ALL0_sreport   YES
* sreport        NO

**/

inline bool U::IsExecutableSiblingPath(const char* argv0,  const char* dirpath )  // static
{
    const char* exename = BaseName_NoSepAsis_(argv0) ; 
    const char* basename = BaseName_NoSepAsis_(dirpath) ;  
    bool is_executable_sibling_path = basename && exename && EndsWith(basename, exename) && strlen(basename) > strlen(exename) ; 

    if(VERBOSE) std::cout 
        << "[U::IsExecutableSiblingPath"
        << std::endl 
        << " argv0 " << ( argv0 ? argv0 : "-" )
        << std::endl 
        << " dirpath " << ( dirpath ? dirpath : "-" )
        << std::endl 
        << " exename " << ( exename ? exename : "-" )
        << std::endl 
        << " basename " << ( basename ? basename : "-" )
        << std::endl 
        << " is_executable_sibling_path " << ( is_executable_sibling_path ? "YES" : "NO " )
        << "]U::IsExecutableSiblingPath"
        << std::endl 
        ;

    return is_executable_sibling_path ; 
}

inline int U::SetEnvDefaultExecutableSiblingPath(const char* ekey, char* argv0, const char* dirpath )
{
    std::string _sibfold = FormExecutableSiblingPath(argv0, dirpath); 
    const char* sibfold = _sibfold.empty() ? nullptr : _sibfold.c_str() ; 
    bool overwrite = false ; 
    int rc = setenvvar( ekey, sibfold, overwrite ); 

    if(VERBOSE) std::cout 
        << "[U::SetEnvDefaultExecutableSiblingPath"
        << std::endl 
        << " ekey " << ( ekey ? ekey : "-" ) 
        << std::endl 
        << " argv0 " << ( argv0 ? argv0 : "-" )
        << std::endl 
        << " ditpath " << ( dirpath ? dirpath : "-" )
        << std::endl 
        << " sibfold " << ( sibfold ? sibfold : "-" )
        << std::endl 
        << " rc " << rc 
        << std::endl 
        << "]U::SetEnvDefaultExecutableSiblingPath"
        << std::endl 
        ;

    return rc ; 
}




/**
U::setenvvar (similar to opticks/sysrap/ssys.h ssys::setenvvar)
-----------------------------------------------------------------

overwrite:false 
    preexisting envvar is not overridden. 

As shell handling of empty strings is inconvenient the special_empty_token char 
allows a single char to represent the empty string, eg '-' 

**/

inline int U::setenvvar( const char* ekey, const char* value, bool overwrite, char special_empty_token)
{
    std::stringstream ss ;
    ss << ekey << "=" ;

    if(value)
    {   
        if(special_empty_token != '\0' && strlen(value) == 1 && value[0] == special_empty_token)
        {   
            ss << "" ; 
        }   
        else
        {   
            ss << value ; 
        }   
    }   

    std::string ekv = ss.str();
    const char* prior = getenv(ekey) ;

    char* ekv_ = const_cast<char*>(strdup(ekv.c_str()));

    int rc = ( overwrite || !prior ) ? putenv(ekv_) : 0  ;   

    const char* after = getenv(ekey) ;

    if(VERBOSE) std::cerr
        << "U::setenvvar"
        << " ekey " << ekey 
        << " ekv " << ekv 
        << " overwrite " << overwrite
        << " prior " << ( prior ? prior : "NULL" )
        << " value " << ( value ? value : "NULL" )   
        << " after " << ( after ? after : "NULL" )   
        << " rc " << rc  
        << std::endl 
        ;   

    return rc ;
}









inline std::string U::FormName_( int idx, int wid )
{ 
    std::stringstream ss ;  
    ss << std::setfill('0') << std::setw(wid) << idx ; 
    std::string s = ss.str(); 
    return s ; 
}
inline const char* U::FormName( int idx, int wid )
{
    std::string s = FormName_(idx, wid) ; 
    return strdup(s.c_str()); 
}


inline std::string U::FormName_( const char* prefix, int idx, const char* ext, int wid )
{
    std::stringstream ss ; 
    if(prefix) ss << prefix ; 

    if( wid > 0 )
    {
        ss << std::setfill('0') << std::setw(wid) << idx ; 
    }
    else
    {
        ss << idx ; 
    }

    if(ext) ss << ext ; 
    std::string s = ss.str(); 
    return s ; 
}
inline const char* U::FormName( const char* prefix, int idx, const char* ext, int wid )
{
    std::string name = FormName_(prefix, idx, ext, wid ); 
    return strdup(name.c_str()); 
}





inline std::string U::FormName_( const char* prefix, const char* body, const char* ext )
{
    std::stringstream ss ; 
    if(prefix) ss << prefix ; 
    if(body) ss << body ; 
    if(ext) ss << ext ; 
    std::string s = ss.str(); 
    return s ; 
}
inline const char* U::FormName( const char* prefix, const char* body, const char* ext )
{
    std::string name = FormName_(prefix, body, ext );  
    return strdup(name.c_str()); 
}

// cctype

inline bool U::isdigit_(char c ) { return std::isdigit(static_cast<unsigned char>(c)) ; }
inline bool U::isalnum_(char c ) { return std::isalnum(static_cast<unsigned char>(c)) ; }
inline bool U::isupper_(char c ) { return std::isupper(static_cast<unsigned char>(c)) ; }
inline bool U::islower_(char c ) { return std::islower(static_cast<unsigned char>(c)) ; }


inline void U::Summarize( std::vector<std::string>& smry_labels, const std::vector<std::string>* labels, int wid )
{
    int num_labels = labels ? labels->size() : 0 ; 
    for(int i=0 ; i < num_labels ; i++)
    {
        const char* label = (*labels)[i].c_str() ; 
        smry_labels.push_back( U::Summarize(label, wid) ) ;  
    }
}

/**
U::Summarize
---------------

Shorten stamp labels via heuristics of distinctive chars    

C++ version of npmeta.py NPMeta::Summarize


* always take first char 
* alnum after _
* upper char following lower 
* accept r or o after P to distinguish Pre and Post 

**/

inline const char* U::Summarize( const char* label, int wid )  // static
{
    if(label == nullptr) return nullptr ;
    std::string smry = U::Summarize_(label, wid); 
    char* _smry = const_cast<char*>(smry.c_str()) ; 
    int len = strlen(_smry) ; 
    if(len > wid) _smry[wid+1] = '\0' ;    
    return strdup(_smry) ; 
}
inline std::string U::Summarize_( const char* label, int wid )  // static
{
    int len = strlen(label) ; 
    std::string str ; 
    if( len <= wid )
    { 
        str = label ; 
    }
    else
    {
        std::stringstream ss ;  
        char p = '\0' ; 
        for(int i=0 ; i < len ; i++) 
        {
           char c = label[i] ; 
           bool take =  ( p == '\0' ) 
                     || ( isalnum_(c) && p == '_' )     
                     || ( isalnum_(c) && p == '_' )
                     || ( isupper_(c) && islower_(p) )
                     || ( p == 'P' && ( c == 'r' || c == 'o' ) )
                     ;
           p = c ;
           if(take) ss << c ;  
        }
        str = ss.str(); 
    }
    return str ; 
}


/**
U::LineVector
--------------



**/

inline void U::LineVector( std::vector<std::string>& lines, const char* LINES )
{
    std::stringstream fss(LINES);
    std::string line ; 
    while(getline(fss, line))
    {
        if(strlen(line.c_str())==0) continue ; 
        lines.push_back(line); 
    }
}


inline void U::LiteralTrim( std::string& line )
{
    const char* trim = " " ;  
    if(strlen(line.c_str())==0) return ; 
    line.erase(0, line.find_first_not_of(trim));  // left trim
    line.erase(line.find_last_not_of(trim) + 1);   // right trim 
}

inline void U::Literal( std::vector<std::string>& lines, const char* LINES )
{
    std::stringstream fss(LINES);
    std::string line ; 
    while(getline(fss, line))
    {
        LiteralTrim(line); 
        if(line.size() == 0) continue ; 
        lines.push_back(line); 
    }
}

inline void U::LiteralAnno( std::vector<std::string>& field, std::vector<std::string>& anno, const char* LINES, const char* delim  )
{
    std::vector<std::string> lines ; 
    U::Literal(lines, LINES ); 
    int num_lines = lines.size();     
    bool dump = false ; 

    if(dump) std::cout << "U::LiteralAnno num_lines " << num_lines << std::endl ; 

    for(int i=0 ; i < num_lines ; i++)
    {
        const std::string& line = lines[i] ; 
        std::size_t pfirst = line.find_first_of(delim) ; // first char matching any of delim char
        std::size_t plast =  line.find_last_of(delim) ;  // last char matching any of delim char

        if(dump) std::cout 
            << " line [" << line << "]" 
            << " pfirst " << pfirst  
            << " plast " << plast  
            << std::endl 
            ;

        if( pfirst != std::string::npos && plast != std::string::npos && plast - pfirst == 1 )
        {
            std::string _field = line.substr(0, pfirst) ; 
            std::string _anno  = line.substr(plast+1) ; 
            LiteralTrim(_field); 
            LiteralTrim(_anno); 
            field.push_back( _field ); 
            anno.push_back( _anno ); 
        } 
        else
        {
            field.push_back(line); 
            anno.push_back(""); 
        }
    }
}





inline std::string U::Space(int wid)
{
    std::stringstream ss ; 
    for(int i=0 ; i < wid ; i++) ss << " " ; 
    std::string str = ss.str(); 
    return str ; 
}



inline std::string U::form_name(const char* stem, const char* ext)
{
    std::stringstream ss ; 
    ss << stem ; 
    ss << ext ; 
    return ss.str(); 
}
inline std::string U::form_path(const char* dir, const char* name)
{
    std::stringstream ss ; 
    ss << dir ; 
    if(name) ss << "/" << name ; 
    return ss.str(); 
}

inline std::string U::form_path(const char* dir, const char* reldir, const char* name)
{
    std::stringstream ss ; 
    ss << dir ; 
    if(reldir) ss << "/" << reldir ; 
    if(name) ss << "/" << name ; 
    return ss.str(); 
}







template<typename ... Args>
std::string U::Path_( Args ... args_  )
{
    std::vector<const char*> args = {args_...} ; 

    std::vector<std::string> elem ; 
    for(unsigned i=0 ; i < args.size() ; i++)
    {   
        const char* arg = args[i] ;   
        if( i == 0 && arg == nullptr ) arg = DEFAULT_PATH_ARG_0 ; 
        if(arg) elem.push_back(arg);  
    }   

    unsigned num_elem = elem.size() ; 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_elem ; i++)
    {   
        const std::string& ele = elem[i] ; 
        ss << ele << ( i < num_elem - 1 ? "/" : "" ) ; 
    }   
    std::string s = ss.str();

    return s ;
}  

template std::string U::Path_( const char*, const char* );
template std::string U::Path_( const char*, const char*, const char* );
template std::string U::Path_( const char*, const char*, const char*, const char* );

template<typename ... Args>
const char* U::Path( Args ... args )
{
    std::string s = Path_(args...)  ;
    return strdup(s.c_str()) ;
}  

template const char* U::Path( const char*, const char* );
template const char* U::Path( const char*, const char*, const char* );
template const char* U::Path( const char*, const char*, const char*, const char* );







#include <cstring>
#include <cstdlib>
#include <cstdio> 
#include <sys/stat.h> 
#include <errno.h> 
#include "dirent.h"

inline int U::MakeDirs( const char* dirpath_, int mode_ )
{
    mode_t default_mode = S_IRWXU | S_IRGRP |  S_IXGRP | S_IROTH | S_IXOTH ;
    mode_t mode = mode_ == 0 ? default_mode : mode_ ;   

    char* dirpath = strdup(dirpath_);
    char* p = dirpath ;   
    int rc = 0 ;   

    while (*p != '\0' && rc == 0)
    {   
        p++;                                 // advance past leading character, probably slash, and subsequent slashes the next line gets to  
        while(*p != '\0' && *p != '/') p++;  // advance p until subsequent slash 
        char v = *p;                         // store the slash      
        *p = '\0' ;                          // replace slash with string terminator
        //printf("%s\n", path );                   
        rc = mkdir(dirpath, mode) == -1 && errno != EEXIST ? 1 : 0 ;  // set rc non-zero for mkdir errors other than exists already  
        *p = v;                              // put back the slash  
    }    
    free(dirpath);
    return rc ;
}

inline int U::MakeDirsForFile( const char* filepath, int mode_ )
{
    if(filepath == nullptr) return 1 ; 
    std::string dirpath = U::DirName(filepath); 
    return MakeDirs(dirpath.c_str(), mode_ );  
}

inline int U::PathType( const char* base, const char* name )
{
    const char* path = Path(base, name); 
    return PathType(path) ; 
}
inline int U::PathType( const char* path )
{
    int rc = ERROR_PATH ; 
    struct stat st ;
    if(0 == stat(path, &st))
    {   
        if(     S_ISDIR(st.st_mode)) rc = DIR_PATH ; 
        else if(S_ISREG(st.st_mode)) rc = FILE_PATH ;   
        else                         rc = OTHER_PATH ; 
    }   
    return rc ;   
}

/**
U::DirList
-----------

**/

inline void U::DirList(std::vector<std::string>& names, const char* path, const char* ext, bool exclude )
{
    DIR* dir = opendir(path) ;
    if(!dir) std::cout << "U::DirList FAILED TO OPEN DIR " << ( path ? path : "-" ) << std::endl ; 
    if(!dir && RAISE) std::raise(SIGINT) ; 
    if(!dir) return ; 
    struct dirent* entry ;
    while ((entry = readdir(dir)) != nullptr) 
    {   
        const char* name = entry->d_name ;
        bool dot_name = strcmp(name,".") == 0 || strcmp(name,"..") == 0 ; 
        if(dot_name) continue ; 
 
        bool ext_match = ext == nullptr ? true : ( strlen(name) > strlen(ext) && strcmp(name + strlen(name) - strlen(ext), ext)==0)  ;
        if(ext_match == true  && exclude == false) names.push_back(name); 
        if(ext_match == false && exclude == true)  names.push_back(name); 
    }   
    closedir (dir);
    std::sort( names.begin(), names.end() );  

    if(names.size() == 0 ) std::cout 
        << "U::DirList" 
        << " path " << ( path ? path : "-" ) 
        << " ext " << ( ext ? ext : "-" ) 
        << " NO ENTRIES FOUND "
        << " exclude " << exclude 
        << std::endl
        ;   
}

inline void U::Trim(std::vector<std::string>& names, const char* ext)
{
    for(int i=0 ; i < int(names.size()) ; i++)
    {   
        std::string& name = names[i]; 
        const char* n = name.c_str();
        bool ends_with_ext =  strlen(n) > strlen(ext)  && strncmp(n + strlen(n) - strlen(ext), ext, strlen(ext) ) == 0 ;
        assert( ends_with_ext );
        name = name.substr(0, strlen(n) - strlen(ext));
    }
}

inline void U::Split(const char* str, char delim,   std::vector<std::string>& elem)
{
    std::stringstream ss; 
    ss.str(str)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(s) ; 
}

inline int U::FindIndex(const std::vector<std::string>& names, const char* name) // static
{
    size_t idx = std::distance( names.begin(), std::find( names.begin(), names.end(), name ));
    return idx >= names.size() ? -1  : int(idx) ;
}





inline std::string U::Desc(const std::vector<std::string>& names)
{
    std::stringstream ss ;
    for(unsigned i=0 ; i < names.size() ; i++) ss << "[" << names[i] << "]" << std::endl ;
    std::string s = ss.str();
    return s ;
}


/**
U::Resolve0 : Old impl that only replaces tokens in first path element
-----------------------------------------------------------------------

::

    $TOKEN/remainder/path/name.npy   (tok_plus) 
    $TOKEN

If the TOKEN envvar is not set then nullptr is returned.  

**/

inline const char* U::Resolve0(const char* spec_, const char* relp_)
{
    if(spec_ == nullptr) return nullptr ; 

    std::string spec_relp = form_path(spec_, relp_); 
    char* spec = strdup(spec_relp.c_str()) ;  

    std::stringstream ss ; 
    if( spec[0] == '$' )
    {   
        char* sep = strchr(spec, '/');       // point to first slash  
        char* end = strchr(spec, '\0' );  
        bool tok_plus =  sep && end && sep != end ;   
        if(tok_plus) *sep = '\0' ;           // replace slash with null termination 
        char* pfx = getenv(spec+1) ; 
        if(pfx == nullptr) return nullptr ; 
        if(tok_plus) *sep = '/' ;            // put back the slash 
        ss << pfx  << ( sep ? sep : "" ) ; 
    }   
    else
    {   
        ss << spec ; 
    }   

    std::string str = ss.str(); 
    const char* path = str.c_str(); 
    return strdup(path) ; 
}

/**
U::Resolve
------------

NB similar to sysrap/spath.h spath::_ResolvePath 

This resolves spec with multiple tokens, eg::

    $HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/jpmt/PMTSimParamData/MPT

Any unresolved token causes nullptr to be returned.

**/

inline const char* U::Resolve(const char* spec_, const char* rel1_, const char* rel2_ )
{
    if(spec_ == nullptr) return nullptr ; 
    std::string spec_relp = form_path(spec_, rel1_, rel2_ ); 
    char* spec = strdup(spec_relp.c_str()) ;  

    std::stringstream ss ; 
    int speclen = int(strlen(spec)) ;  
    char* end = strchr(spec, '\0' ); 
    int i = 0 ; 

    if(VERBOSE) std::cout << " spec " << spec << " speclen " << speclen << std::endl ; 

    while( i < speclen )
    {
        if(VERBOSE) std::cout << " i " << i << " spec[i] " << spec[i] << std::endl ;   
        if( spec[i] == '$' )
        {
            char* p = spec + i ; 
            char* sep = strchr( p, '/' ) ; // first slash after token   
            bool tok_plus =  sep && end && sep != end ;  
            if(tok_plus) *sep = '\0' ;           // replace slash with null termination 
            char* val = getenv(p+1) ;  // skip '$'
            int toklen = int(strlen(p)) ;  // strlen("TOKEN")  no need for +1 as already at '$'  
            if(VERBOSE) std::cout << " toklen " << toklen << std::endl ;  
            if(val == nullptr) 
            {
                std::cerr 
                    << "U::Resolve token [" 
                    << p+1 
                    << "] does not resolve " 
                    << std::endl 
                    ; 
                return nullptr ;    // all tokens must resolve 
            }
            if(tok_plus) *sep = '/' ;            // put back the slash 
            ss << val  ; 

            i += toklen ;   // skip over the token 
        }
        else
        {
           ss << spec[i] ; 
           i += 1 ; 
        }
    }
    std::string str = ss.str(); 
    const char* path = str.c_str(); 
    return strdup(path) ; 
}









inline void U::WriteString( const char* dir, const char* reldir, const char* name, const char* str )  // static 
{
    std::string path = form_path(dir, reldir, name); 
    WriteString(path.c_str(), str); 
}

inline void U::WriteString( const char* dir, const char* name, const char* str )  // static 
{
    std::string path = form_path(dir, name); 
    WriteString(path.c_str(), str); 
}

inline void U::WriteString( const char* path, const char* str )  // static 
{
    if(str == nullptr) return ; 
    std::ofstream fp(path, std::ios::out);
    fp << str ; 
    fp.close(); 
}


inline const char* U::ReadString( const char* dir, const char* reldir, const char* name) // static
{
    std::string path = form_path(dir, reldir, name); 
    return ReadString(path.c_str()); 
}

inline const char* U::ReadString( const char* dir, const char* name) // static
{
    std::string path = form_path(dir, name); 
    return ReadString(path.c_str()); 
}

inline const char* U::ReadString( const char* path_ )  // static
{
    const char* path = Resolve(path_); 
    std::vector<std::string> lines ; 
    std::string line ; 
    std::ifstream ifs(path); 
    while(std::getline(ifs, line)) lines.push_back(line); 
    unsigned num_lines = lines.size(); 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_lines ; i++)
    {
        ss << lines[i] ; 
        if( i < num_lines - 1 ) ss << std::endl ; 
    }
    std::string str = ss.str(); 
    return str.empty() ? nullptr : strdup(str.c_str()) ; 
}

inline const char* U::ReadString2(const char* path_)  // static
{
    const char* path = Resolve(path_); 
    std::ifstream ifs(path);
    std::stringstream ss ;
    ss << ifs.rdbuf();
    std::string str = ss.str(); 
    return str.empty() ? nullptr : strdup(str.c_str()) ; 
}


inline uint64_t U::Now() // static
{
    // from opticks/sysrap/sstamp.h 
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds ;
    std::chrono::time_point<Clock> t0 = Clock::now();
    return std::chrono::duration_cast<Unit>(t0.time_since_epoch()).count() ;   
}

/**
U::LooksLikeStampInt
----------------------

Contemporary microsecond uint64_t timestamps look like below with 16 digits::

    1700224486350245

::

    In [20]: np.c_[np.array([0,int(1e15),1700224486350245,int(1e16),int(0x7ffffffffffffff) ]).view("datetime64[us]")]
    Out[20]:
    array([[ '1970-01-01T00:00:00.000000'],
           [ '2001-09-09T01:46:40.000000'],
           [ '2023-11-17T12:34:46.350245'],
           [ '2286-11-20T17:46:40.000000'],
           ['20237-04-25T10:45:03.423487']], dtype='datetime64[us]')

**/

inline bool U::LooksLikeStampInt(const char* str) // static
{   
    int length = strlen(str) ;
    int digits = 0 ; 
    for(int i=0 ; i < length ; i++) if(str[i] >= '0' && str[i] <= '9') digits += 1 ;
    return length == 16 && digits == length  ;
}


template<typename T>
inline bool U::LooksLikeTimestamp( T value )
{
    return sizeof(T) == 8 && value > 1700000000000000 ; 
}


/**
U::LooksLikeProfileTriplet
-----------------------------

Follows sprof::LooksLikeProf, repeated hear for convenience

**/

inline bool U::LooksLikeProfileTriplet(const char* str) // static
{
    int len = str ? int(strlen(str)) : 0 ; 
    int count_delim = 0 ; 
    int count_non_digit = 0 ; 
    int first_field_digits = 0 ; 

    for(int i=0 ; i < len ; i++ ) 
    {   
        char c = str[i] ; 
        bool is_digit = c >= '0' && c <= '9' ;
        bool is_delim = c == ',' ; 
        if(!is_digit) count_non_digit += 1 ; 
        if(count_delim == 0 && is_digit ) first_field_digits += 1 ;   
        if(is_delim) count_delim += 1 ; 
    }   
    bool heuristic = count_delim == 2 && count_non_digit == count_delim && first_field_digits == 16 ; 
    return heuristic ;    
}



inline std::string U::Format(uint64_t t, const char* fmt) // static
{
    // from opticks/sysrap/sstamp.h 
    if(t == 0) t = Now() ; 
    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds  ;   
    std::chrono::time_point<Clock> tp{Unit{t}} ; 

    std::time_t tt = Clock::to_time_t(tp);

    // extract the sub second part from the duration since epoch
    auto subsec = std::chrono::duration_cast<Unit>(tp.time_since_epoch()) % std::chrono::seconds{1};

    std::stringstream ss ; 
    ss  
       << std::put_time(std::localtime(&tt), fmt ) 
       << std::setfill('0') 
       << std::setw(6) << subsec.count() 
       ;   

    std::string str = ss.str(); 
    return str ; 
}

inline std::string U::FormatInt(int64_t t, int wid ) // static
{
    std::stringstream ss ; 
    ss.imbue(std::locale("")) ;  // commas for thousands

    if( t > -1 ) ss << std::setw(wid) << t ;  
    else         ss << std::setw(wid) << "" ; 
    std::string str = ss.str(); 
    return str ; 
}

/**
U::LastDigit
-------------

Start from the end of the string, returning when reach 
first digit from end (aka last digit from front). 

**/

inline char* U::LastDigit(const char* str)
{
    if(str == nullptr) return nullptr ; 
    char* s = const_cast<char*>(str); 
    char* p = s+strlen(s)-1 ; 
    while( p > s )
    {
       if( *p >= '0' && *p <= '9' ) break ; 
       p-- ; 
    }
    return p ; 
}

inline char* U::FirstDigit(const char* str)
{
    if(str == nullptr) return nullptr ; 
    char* s = const_cast<char*>(str); 
    char* p = s ; 
    while( p )
    {
       if( *p >= '0' && *p <= '9' ) break ; 
       p++ ; 
    }
    return p ; 
}

inline char* U::FirstToLastDigit(const char* str)
{
    const char* s = strdup(str); 
    char* f = FirstDigit(s); 
    char* l = LastDigit(s); 
    char* r = nullptr ; 
    if(f && l && l + 1  > f)
    {
        if(*(l+1) != '\0') *(l+1) = '\0' ; 
        r = f ; 
    }
    return r ; 
}














struct NPU
{
    static constexpr char* MAGIC = (char*)"\x93NUMPY" ; 
    static constexpr bool  FORTRAN_ORDER = false ;

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

    static std::string _make_descr(bool little_endian, char uifc, int width );
    static std::string _make_narrow(const char* descr);  
    static std::string _make_wide(const char* descr);  
    static std::string _make_other(const char* descr, char other);  

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

template<typename T>
inline std::string NPU::make_header(const std::vector<int>& shape )
{
    //std::string descr = Desc<T>::descr() ;   
    std::string descr = descr_<T>::dtype() ;

    return _make_header( shape, descr.c_str() ) ; 
}

template<typename T>
inline std::string NPU::make_jsonhdr(const std::vector<int>& shape )
{
    //std::string descr = Desc<T>::descr() ; 
    std::string descr = descr_<T>::dtype() ;
    return _make_jsonhdr( shape, descr.c_str() ) ; 
}

inline std::string NPU::xxdisplay(const std::string& hdr, int width, char non_printable)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < hdr.size() ; i++) 
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

    // previously used "char" here,
    // but thats a bug as when the header exceeds 128 bytes 
    // it flips -ve (twos complement)
    // observed this first with the bnd.npy which has 5 dimensions
    // causing the header to be larger than for example icdf.npy with 3 dimensions
 
    unsigned char hlen_lsb = hdr[8] ;  
    unsigned char hlen_msb = hdr[9] ;  
    int hlen = hlen_msb << 8 | hlen_lsb ; 

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
    assert( hlen > 0 ); 
    assert( (hlen+10) % 16 == 0 ) ;  
    assert( hlen+10 == int(hdr.size()) ) ; 

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
    for(unsigned i=0 ; i < strlen(dict) ; i++)
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

inline std::string NPU::_make_descr(bool little_endian, char uifc, int width ) // static
{
    std::stringstream ss ; 
    ss << ( little_endian ? '<' : '>' ) << uifc << width ;  
    return ss.str(); 
}

inline std::string NPU::_make_narrow(const char* descr) // static
{
    bool little_endian ; 
    char uifc ; 
    int ebyte ; 
    _parse_descr( little_endian, uifc, ebyte, descr ); 
    return _make_descr(little_endian, uifc, ebyte/2  ); 
} 

inline std::string NPU::_make_wide(const char* descr) // static
{
    bool little_endian ; 
    char uifc ; 
    int ebyte ; 
    _parse_descr( little_endian, uifc, ebyte, descr ); 
    return _make_descr(little_endian, uifc, ebyte*2  ); 
} 


inline std::string NPU::_make_other(const char* descr, char other) // static
{
    bool little_endian ; 
    char uifc ; 
    int ebyte ; 
    _parse_descr( little_endian, uifc, ebyte, descr ); 
    return _make_descr(little_endian, other, ebyte  ); 
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

    union u16c2_t { 
        uint16_t u16 ; 
        char     c[2] ;  
    }; 

    u16c2_t len ; 
    len.u16 = dlen ; 

    char e = endian::detect() ; 
    std::string hlen(2, ' ') ;
    hlen[0] = e == endian::LITTLE ? len.c[0] : len.c[1] ;  
    hlen[1] = e == endian::LITTLE ? len.c[1] : len.c[0] ; 

#ifdef NPU_DEBUG
    std::cout << " dlen " << dlen << std::endl ; 
    std::cout << " len.c[0] " << len.c[0] << std::endl ; 
    std::cout << " len.c[1] " << len.c[1] << std::endl ; 
    std::cout << ( e == endian::LITTLE ? "little_endian" : "big_endian" ) << std::endl ; 
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



/**
nview.h
=========

Templated reinterpretation of bits allowing to view 
unsigned int as float and double and vice versa.

Note this is present in sysrap/sview.h 

**/

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define NVIEW_METHOD __host__ __device__ __forceinline__
#else
#    define NVIEW_METHOD inline
#endif

struct nview
{
    union UIF32 
    {
        unsigned u ;  
        int i  ;  
        float f ; 
    }; 

    struct uint2 { unsigned x, y ; }; 
    struct int2  { unsigned x, y ; }; 

    union UIF64
    {
        uint2  uu ;  
        int2   ii ;  
        double f ; 
    }; 

    template<typename T> static T int_as( int i ); 
    template<typename T> static int int_from( T v ); 

    template<typename T> static T uint_as( unsigned u ); 
    template<typename T> static unsigned uint_from( T v ); 
}; 

template<> NVIEW_METHOD float nview::int_as<float>( int i )
{
     UIF32 u32 ; 
     u32.i = i ; 
     return u32.f ; 
}
template<> NVIEW_METHOD double nview::int_as<double>( int i )
{
     UIF64 u64 ; 
     u64.ii.x = i ; 
     return u64.f ; 
}


template<> NVIEW_METHOD int nview::int_from<float>( float f )
{
     UIF32 u32 ; 
     u32.f = f ; 
     return u32.i ; 
}
template<> NVIEW_METHOD int nview::int_from<double>( double f )
{
     UIF64 u64 ; 
     u64.f = f ; 
     return u64.ii.x  ; 
}

template<> NVIEW_METHOD float nview::uint_as<float>( unsigned u )
{
     UIF32 u32 ; 
     u32.u = u ; 
     return u32.f ; 
}
template<> NVIEW_METHOD double nview::uint_as<double>( unsigned v )
{
     UIF64 u64 ; 
     u64.uu.x = v ; 
     return u64.f ; 
}

template<> NVIEW_METHOD unsigned nview::uint_from<float>( float f )
{
     UIF32 u32 ; 
     u32.f = f ; 
     return u32.u ; 
}
template<> NVIEW_METHOD unsigned nview::uint_from<double>( double f )
{
     UIF64 u64 ; 
     u64.f = f ; 
     return u64.uu.x ; 
}






struct UName
{
    static constexpr const char* UNSET = "UNSET" ; 
    std::vector<std::string> names ;  
    int count ; 

    UName(); 
    int get(const char* name) ; 
    int add(const char* name, bool dump=false) ; 
    std::string desc() const ; 
    std::string as_str() const ; 
}; 


inline UName::UName()
    :
    count(0)
{
    names.push_back(UNSET); 
}

inline int UName::add(const char* name, bool dump)
{
    if(std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name) ; 
    int idx = get(name) ; 
    if(dump) std::cerr 
        << "UName::add"
        << " idx " << std::setw(4) << idx
        << " name " << name
        << " count " << std::setw(5) << count 
        << " size " << std::setw(5) << names.size()
        << std::endl 
        ;

    //if(dump) std::cerr << desc() ; 
    count += 1 ; 
    return idx ;  
}
inline int UName::get(const char* name)
{
    size_t idx = std::distance( names.begin(), std::find(names.begin(), names.end(), name) ) ; 
    size_t num = names.size() ; 
    std::cout << "UName::get " << ( name ? name : "-" ) << " num " << num << " idx " << idx << std::endl ; 
    return idx == num ? -1 : idx ; 
} 
inline std::string UName::desc() const
{
    std::stringstream ss ; 
    ss << "UName::desc" 
       << " count " << std::setw(5) << count 
       << " size " << std::setw(5) << names.size()
       << std::endl
       ; 

    for(int i=0 ; i < int(names.size()) ; i++ ) ss << std::setw(5) << i << " : " << names[i] << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}
inline std::string UName::as_str() const
{
    int num_names = names.size(); 
    std::stringstream ss ; 
    for(int i=0 ; i < num_names ; i++ ) 
    {
        ss << names[i] ; 
        if( i < num_names - 1 ) ss << std::endl ; 
    }
    std::string str = ss.str(); 
    return str ; 
}





union uc4 
{
    char c[4] ; 
    unsigned u ; 

    void set(const char* s4) ; 
    std::string get() const ; 
    std::string desc() const ; 
}; 

inline void uc4::set(const char* s4)
{
    for(unsigned i=0 ; i < 4 ; i++) c[i] = i < strlen(s4) ? s4[i] : '\0' ;  
}
inline std::string uc4::get() const 
{
    const char* p = &c[0] ; 
    std::string str(p, p+4) ; 
    return str ; 
}
inline std::string uc4::desc() const 
{
    std::stringstream ss ; 
    ss << get() << " " << u  ; 
    std::string str = ss.str(); 
    return str ; 
}






union uc8
{
    char     c[8] ; 
    uint64_t u ; 

    void set(const char* s8) ; 
    std::string get() const ; 
    std::string desc() const ; 
}; 
inline void uc8::set(const char* s8)
{
    for(unsigned i=0 ; i < 8 ; i++) c[i] = i < strlen(s8) ? s8[i] : '\0' ;  
}
inline std::string uc8::get() const 
{
    const char* p = &c[0] ; 
    std::string str(p, p+8) ; 
    return str ; 
}
inline std::string uc8::desc() const 
{
    std::stringstream ss ; 
    ss << get() << " " << u  ; 
    std::string str = ss.str(); 
    return str ; 
}



#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>

#include "NPU.hh"

/**
NP 
===

This is developed in https://github.com/simoncblyth/np/
but given the header-only nature is often just incorporated into 
other projects together with NPU.hh

NP.hh(+NPU.hh) provides lightweight header only NPY writing/reading. 
Just copy into your project and ``#include "NP.hh"`` to use. 


**/

struct NP
{
    NP(const char* dtype_="<f4", int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    void set_dtype(const char* dtype_); // may change shape and size of array while retaining the same underlying bytes 

    static void sizeof_check(); 
    static NP* Load(const char* path); 
    static NP* Load(const char* dir, const char* name); 
    static NP* MakeDemo(const char* dtype="<f4" , int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 

    template<typename T> static void Write(const char* dir, const char* name, const std::vector<T>& values ); 
    template<typename T> static void Write(const char* dir, const char* name, const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    static void WriteNames(const char* dir, const char* name, const std::vector<std::string>& names, unsigned num_names=0 ); 


    template<typename T> T*       values() ; 
    template<typename T> const T* values() const  ; 
    template<typename T> void fill(T value); 
    template<typename T> void _fillIndexFlat(T offset=0); 
    template<typename T> void _dump(int i0=-1, int i1=-1) const ;   
    template<typename T> std::string _present(T v) const ; 

    void fillIndexFlat(); 
    void dump(int i0=-1, int i1=-1) const ; 


    int load(const char* path);   
    int load(const char* dir, const char* name);   
    static std::string form_path(const char* dir, const char* name);   

    void save_header(const char* path);   
    void save(const char* path);   
    void save(const char* dir, const char* name);   


    std::string get_jsonhdr_path() const ; // .npy -> .npj on loaded path
    void save_jsonhdr();    
    void save_jsonhdr(const char* path);   
    void save_jsonhdr(const char* dir, const char* name);   

    std::string desc() const ; 

    char*       bytes();  
    const char* bytes() const ;  

    unsigned num_values() const ; 
    unsigned arr_bytes() const ;   // formerly num_bytes
    unsigned hdr_bytes() const ;  
    unsigned meta_bytes() const ;
  
    // primary data members 
    std::vector<char> data ; 
    std::vector<int>  shape ; 
    std::string       meta ; 

    // transient
    std::string lpath ; 

    // headers used for transport 
    std::string _hdr ; 
    std::string _prefix ; 

    // results from parsing _hdr or set_dtype 
    const char* dtype ; 
    char        uifc ;    // element type code 
    int         ebyte ;   // element bytes  
    int         size ;    // number of elements from shape


    void        update_headers();     

    std::string make_header() const ; 
    bool        decode_header() ; // sets shape based on arr header

    std::string make_prefix() const ; 
    bool        decode_prefix() ; // also resizes buffers ready for reading in 
    unsigned    prefix_size(unsigned index) const ;     

    std::string make_jsonhdr() const ;
};

/**
operator<< NP : NOT a member function
---------------------------------------

Write array into output stream 

**/

inline std::ostream& operator<<(std::ostream &os,  const NP& a) 
{ 
    os << a.make_prefix() ; 
    os << a.make_header() ; 
    os.write(a.bytes(), a.arr_bytes());
    os << a.meta ; 
    return os ; 
}

/**
operator>> NP : NOT a member function
---------------------------------------

Direct input stream into NP array

**/

inline std::istream& operator>>(std::istream& is, NP& a)     
{
    is.read( (char*)a._prefix.data(), net_hdr::LENGTH ) ; 

    unsigned hdr_bytes_nh = a.prefix_size(0); 
    unsigned arr_bytes_nh = a.prefix_size(1); 
    unsigned meta_bytes_nh = a.prefix_size(2);  

    std::cout 
        << " hdr_bytes_nh " << hdr_bytes_nh 
        << " arr_bytes_nh " << arr_bytes_nh 
        << " meta_bytes_nh " << meta_bytes_nh 
        << std::endl
        ;

    std::getline( is, a._hdr );  
    a._hdr += '\n' ;     // getline consumes newline ending header but does not return it 
    assert( hdr_bytes_nh == a._hdr.length() ); 

    a.decode_header();   // resizes data array 

    assert( a.arr_bytes() == arr_bytes_nh ); 
    is.read(a.bytes(), a.arr_bytes() );

    a.meta.resize(meta_bytes_nh);
    is.read( (char*)a.meta.data(), meta_bytes_nh );
 
    //is.setstate(std::ios::failbit);
    return is;
}



/**
NP::update_headers
-------------------

Updates network header "prefix" and array header descriptions of the object.

Cannot do this automatically in setters that change shape, dtype or metadata
because are using a struct.  So just have to invoke this before streaming.  

**/

inline void NP::update_headers()
{
    std::string net_hdr = make_prefix(); 
    _prefix.assign(net_hdr.data(), net_hdr.length()); 

    std::string hdr =  make_header(); 
    _hdr.resize(hdr.length()); 
    _hdr.assign(hdr.data(), hdr.length()); 
}


inline std::string NP::make_header() const 
{
    std::string hdr =  NPU::_make_header( shape, dtype ) ;
    return hdr ; 
}
inline std::string NP::make_prefix() const 
{
    std::vector<unsigned> parts ;
    parts.push_back(hdr_bytes());
    parts.push_back(arr_bytes());
    parts.push_back(meta_bytes());
    parts.push_back(0);    // xxd neater to have 16 byte prefix 

    std::string net_hdr = net_hdr::pack( parts ); 
    return net_hdr ; 
}
inline std::string NP::make_jsonhdr() const 
{
    std::string json = NPU::_make_jsonhdr( shape, dtype ) ; 
    return json ; 
}  


inline NP* NP::MakeDemo(const char* dtype, int ni, int nj, int nk, int nl, int nm )
{
    NP* a = new NP(dtype, ni, nj, nk, nl, nm);
    a->fillIndexFlat(); 
    return a ; 
}




/**
NP::decode_prefix
-------------------

This is used for boost asio handlers to resize the 
object buffers as directed by the sizes extracted 
from the prefix header. For example see:: 

   np_client::handle_read_header
   np_session::handle_read_header

Note that this is not used when streaming in 
from file. There is no prefix header in that 
situation.

**/
inline bool NP::decode_prefix()  
{
    unsigned hdr_bytes_nh = prefix_size(0); 
    unsigned arr_bytes_nh = prefix_size(1); 
    unsigned meta_bytes_nh = prefix_size(2);  

    std::cout 
        << "NP::decode_prefix"
        << " hdr_bytes_nh " << hdr_bytes_nh
        << " arr_bytes_nh " << arr_bytes_nh
        << " meta_bytes_nh " << meta_bytes_nh
        << std::endl
        ;

    bool valid = hdr_bytes_nh > 0 ; 
    if(valid)
    {
        _hdr.resize(hdr_bytes_nh);
        data.resize(arr_bytes_nh);   // data now vector of chars 
        meta.resize(meta_bytes_nh);
    }
    return valid ; 
}
inline unsigned NP::prefix_size(unsigned index) const { return net_hdr::unpack(_prefix, index); }  

/**
NP::decode_header
-----------------------

Array header _hdr is parsed setting the below and data is resized.

shape
    vector of int 
uifc
    element type code
ebyte 
    element number of bytes
size
    number of elements 

Decoding the header gives the shape of the
data, so together with the size of the type
know how many bytes can read from the remainder of the stream
following the header.

**/

inline bool NP::decode_header()  
{
    shape.clear(); 
    std::string descr ; 
    NPU::parse_header( shape, descr, uifc, ebyte, _hdr ) ; 
    dtype = strdup(descr.c_str());  
    size = NPS::size(shape);    // product of shape dimensions 
    data.resize(size*ebyte) ;   // data is now just char 
    return true  ; 
}


/**
NP::set_dtype
--------------

Setting a dtype with a different element size ebyte 
necessarily changes shape and size of array.

CAUTION this will cause asserts if the array shape is such 
that the dtype change and resulting shape change would 
change the total number of bytes in the array.

**/

inline void NP::set_dtype(const char* dtype_)
{
    char uifc_ = NPU::_dtype_uifc(dtype_) ; 
    int  ebyte_ = NPU::_dtype_ebyte(dtype_) ; 
    assert( ebyte_ == 1 || ebyte_ == 2 || ebyte_ == 4 || ebyte_ == 8 ); 

    std::cout 
        << "changing dtype/uifc/ebyte from: " 
        << dtype << "/" << uifc << "/" << ebyte 
        << " to: "
        << dtype_ << "/" << uifc_ << "/" << ebyte_ 
        << std::endl
        ;          

    if( ebyte_ == ebyte )
    {
        std::cout << "NP::set_dtype : no change in ebyte keeps same array dimensions" << std::endl ; 
    }
    else if( ebyte_ < ebyte )
    {
        int expand = ebyte/ebyte_ ; 
        std::cout << "NP::set_dtype : shifting to smaller ebyte increases array dimensions, expand: " << expand << std::endl ; 
        for(unsigned i=0 ; i < shape.size() ; i++ ) shape[i] *= expand ; 
    }
    else if( ebyte_ > ebyte )
    {
        int shrink = ebyte_/ebyte ; 
        std::cout << "NP::set_dtype : shifting to larger ebyte decreases array dimensions, shrink: " << shrink << std::endl ; 
        for(unsigned i=0 ; i < shape.size() ; i++ ) shape[i] /= shrink  ; 
    }

    int num_bytes  = size*ebyte ;      // old 
    int size_ = NPS::size(shape) ;     // new
    int num_bytes_ = size_*ebyte_ ;    // new 

    bool allowed_change = num_bytes_ == num_bytes ; 
    if(!allowed_change)
    {
        std::cout << "NP::set_dtype : NOT ALLOWED as it would change the number of bytes " << std::endl ; 
        std::cout << " old num_bytes " << num_bytes << " proposed num_bytes_ " << num_bytes_ << std::endl ;   
    }
    assert( allowed_change ); 

    // change the members

    dtype = strdup(dtype_); 
    uifc  = uifc_ ;   
    ebyte = ebyte_ ; 
    size = size_ ; 

    std::cout << desc() << std::endl ; 
}


// former approach assumed data is already sized : but shape comes first 

inline unsigned NP::hdr_bytes() const { return _hdr.length() ; }
inline unsigned NP::num_values() const { return NPS::size(shape) ;  }
inline unsigned NP::arr_bytes()  const { return NPS::size(shape)*ebyte ; }
inline unsigned NP::meta_bytes() const { return meta.length() ; }

inline char*        NP::bytes() { return (char*)data.data() ;  } 
inline const char*  NP::bytes() const { return (char*)data.data() ;  } 


inline NP::NP(const char* dtype_, int ni, int nj, int nk, int nl, int nm )
    :
    dtype(strdup(dtype_)),
    uifc(NPU::_dtype_uifc(dtype)),
    ebyte(NPU::_dtype_ebyte(dtype)),
    size(NPS::set_shape(shape, ni,nj,nk,nl,nm ))
{
    data.resize( size*ebyte ) ;  // vector of char  
    std::fill( data.begin(), data.end(), 0 );     

    _prefix.assign(net_hdr::LENGTH, '\0' ); 
    _hdr = make_header(); 
}




template<typename T> inline T*  NP::values() { return (T*)data.data() ;  } 

template<typename T> inline void NP::fill(T value)
{
    T* vv = values<T>(); 
    for(unsigned i=0 ; i < size ; i++) *(vv+i) = value ; 
}

template<typename T> inline void NP::_fillIndexFlat(T offset)
{
    T* vv = values<T>(); 
    for(unsigned i=0 ; i < size ; i++) *(vv+i) = T(i) + offset ; 
}


inline void NP::sizeof_check() // static 
{
    assert( sizeof(float) == 4  );  
    assert( sizeof(double) == 8  );  

    assert( sizeof(char) == 1 );  
    assert( sizeof(short) == 2 );
    assert( sizeof(int)   == 4 );
    assert( sizeof(long)  == 8 );
    assert( sizeof(long long)  == 8 );
}

inline void NP::fillIndexFlat()
{
    if(uifc == 'f')
    {   
        switch(ebyte)
        {   
            case 4: _fillIndexFlat<float>()  ; break ; 
            case 8: _fillIndexFlat<double>() ; break ; 
        }   
    }   
    else if(uifc == 'u')
    {   
        switch(ebyte)
        {   
            case 1: _fillIndexFlat<unsigned char>()  ; break ; 
            case 2: _fillIndexFlat<unsigned short>()  ; break ; 
            case 4: _fillIndexFlat<unsigned int>() ; break ; 
            case 8: _fillIndexFlat<unsigned long>() ; break ; 
        }   
    }   
    else if(uifc == 'i')
    {   
        switch(ebyte)
        {   
            case 1: _fillIndexFlat<char>()  ; break ; 
            case 2: _fillIndexFlat<short>()  ; break ; 
            case 4: _fillIndexFlat<int>() ; break ; 
            case 8: _fillIndexFlat<long>() ; break ; 
        }   
    }   
}


inline void NP::dump(int i0, int i1) const 
{
    if(uifc == 'f')
    {   
        switch(ebyte)
        {   
            case 4: _dump<float>(i0,i1)  ; break ; 
            case 8: _dump<double>(i0,i1) ; break ; 
        }   
    }   
    else if(uifc == 'u')
    {   
        switch(ebyte)
        {   
            case 1: _dump<unsigned char>(i0,i1)  ; break ; 
            case 2: _dump<unsigned short>(i0,i1)  ; break ; 
            case 4: _dump<unsigned int>(i0,i1) ; break ; 
            case 8: _dump<unsigned long>(i0,i1) ; break ; 
        }   
    }   
    else if(uifc == 'i')
    {   
        switch(ebyte)
        {   
            case 1: _dump<char>(i0,i1)  ; break ; 
            case 2: _dump<short>(i0,i1)  ; break ; 
            case 4: _dump<int>(i0,i1) ; break ; 
            case 8: _dump<long>(i0,i1) ; break ; 
        }   
    }   
}

inline std::string NP::desc() const 
{
    std::stringstream ss ; 
    ss << "NP " 
       << " dtype " << dtype
       << NPS::desc(shape) 
       << " size " << size 
       << " uifc " << uifc 
       << " ebyte " << ebyte 
       << " shape.size " << shape.size() 
       << " data.size " << data.size()
       << " meta.size " << meta.size() 
       ;
    return ss.str(); 
}

inline NP* NP::Load(const char* path)
{
    NP* a = new NP() ; 
    int rc = a->load(path) ; 
    return rc == 0 ? a  : NULL ; 
}

inline NP* NP::Load(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    return Load(path.c_str());
}

inline std::string NP::form_path(const char* dir, const char* name)
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    return ss.str(); 
}

inline int NP::load(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    return load(path.c_str()); 
}

/**
NP::load
----------

Formerly read an arbitrary initial buffer size, 
now reading up to first newline, which marks the 
end of the header, then adding the newline to the 
header string for correctness as getline consumes the 
newline from the stream without returning it. 

**/

inline int NP::load(const char* path)
{
    lpath = path ;  // loadpath 

    std::ifstream fp(path, std::ios::in|std::ios::binary);
    if(fp.fail())
    {
        std::cerr << "Failed to load from path " << path << std::endl ; 
        return 1 ; 
    }

    std::getline(fp, _hdr );   
    _hdr += '\n' ; 

    decode_header(); 

    fp.read(bytes(), arr_bytes() );

    return 0 ; 
}

inline void NP::save_header(const char* path)
{
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
}

inline void NP::save(const char* path)
{
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
    stream.write( bytes(), arr_bytes() );
}

inline void NP::save(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    save(path.c_str()); 
}

inline void NP::save_jsonhdr(const char* path)
{
    std::string json = make_jsonhdr(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << json ; 
}

inline void NP::save_jsonhdr(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    save_jsonhdr(path.c_str()); 
}

inline std::string NP::get_jsonhdr_path() const 
{
    assert( lpath.empty() == false ); 
    assert( U::EndsWith(lpath.c_str(), ".npy" ) ); 
    std::string path = U::ChangeExt(lpath.c_str(), ".npy", ".npj"); 
    return path ; 
}

inline void NP::save_jsonhdr()
{
    std::string path = get_jsonhdr_path() ; 
    std::cout << "NP::save_jsonhdr to " << path << std::endl  ; 
    save_jsonhdr(path.c_str()); 
}


template <typename T> inline std::string NP::_present(T v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << v  ;      
    return ss.str();
}

// needs specialization to _present char as an int rather than a character
template<>  inline std::string NP::_present(char v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << int(v)  ;      
    return ss.str();
}
template<>  inline std::string NP::_present(unsigned char v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << unsigned(v)  ;      
    return ss.str();
}
template<>  inline std::string NP::_present(float v) const
{
    std::stringstream ss ; 
    ss << " " << std::setw(10) << std::fixed << std::setprecision(3) << v ;
    return ss.str();
}
template<>  inline std::string NP::_present(double v) const
{
    std::stringstream ss ; 
    ss << " " << std::setw(10) << std::fixed << std::setprecision(3) << v ;
    return ss.str();
}


template <typename T> inline void NP::_dump(int i0_, int i1_) const 
{
    int ni = NPS::ni_(shape) ;
    int nj = NPS::nj_(shape) ;
    int nk = NPS::nk_(shape) ;

    int i0 = i0_ == -1 ? 0                : i0_ ;  
    int i1 = i1_ == -1 ? std::min(ni, 10) : i1_ ;  

    std::cout 
       << desc() 
       << std::endl 
       << " array dimensions " 
       << " ni " << ni 
       << " nj " << nj 
       << " nk " << nk
       << " item range i0:i1 "
       << " i0 " << i0 
       << " i1 " << i1 
       << std::endl 
       ;  

    const T* vv = values<T>(); 

    for(int i=i0 ; i < i1 ; i++){
        std::cout << "[" << i  << "] " ;
        for(int j=0 ; j < nj ; j++){
            for(int k=0 ; k < nk ; k++)
            {
                int index = i*nj*nk + j*nk + k ; 
                T v = *(vv + index) ; 
                std::cout << _present<T>(v)  ;      
            }
            //std::cout << std::endl ; 
        }
        std::cout << std::endl ; 
    }


    std::cout 
        << "meta:[" 
        << std::endl
        << meta
        << std::endl
        << "]"
        << std::endl
        ; 
}


/**

specialize-head(){ cat << EOH
// template specializations generated by below bash functions

EOH
}

specialize-types(){ cat << EOT
float
double
char
short
int
long
long long
unsigned char
unsigned short
unsigned int
unsigned long
unsigned long long
EOT
}

specialize-(){
    cat << EOC | perl -pe "s,T,$1,g" - 
template<> inline const T* NP::values<T>() const { return (T*)data.data() ; }
template<> inline       T* NP::values<T>()      {  return (T*)data.data() ; }
template   void NP::_fillIndexFlat<T>(T) ;

EOC
}
specialize(){ specialize-head ; specialize-types | while read t ; do specialize- "$t" ; done  ; }
specialize

**/


// template specializations generated by below bash functions

template<>  inline const float* NP::values<float>() const { return (float*)data.data() ; }
template<>  inline       float* NP::values<float>()      {  return (float*)data.data() ; }
template    void NP::_fillIndexFlat<float>(float) ;

template<> inline const double* NP::values<double>() const { return (double*)data.data() ; }
template<> inline       double* NP::values<double>()      {  return (double*)data.data() ; }
template   void NP::_fillIndexFlat<double>(double) ;

template<> inline const char* NP::values<char>() const { return (char*)data.data() ; }
template<> inline       char* NP::values<char>()      {  return (char*)data.data() ; }
template   void NP::_fillIndexFlat<char>(char) ;

template<> inline const short* NP::values<short>() const { return (short*)data.data() ; }
template<> inline       short* NP::values<short>()      {  return (short*)data.data() ; }
template   void NP::_fillIndexFlat<short>(short) ;

template<> inline const int* NP::values<int>() const { return (int*)data.data() ; }
template<> inline       int* NP::values<int>()      {  return (int*)data.data() ; }
template   void NP::_fillIndexFlat<int>(int) ;

template<> inline const long* NP::values<long>() const { return (long*)data.data() ; }
template<> inline       long* NP::values<long>()      {  return (long*)data.data() ; }
template   void NP::_fillIndexFlat<long>(long) ;

template<> inline const long long* NP::values<long long>() const { return (long long*)data.data() ; }
template<> inline       long long* NP::values<long long>()      {  return (long long*)data.data() ; }
template   void NP::_fillIndexFlat<long long>(long long) ;

template<> inline const unsigned char* NP::values<unsigned char>() const { return (unsigned char*)data.data() ; }
template<> inline       unsigned char* NP::values<unsigned char>()      {  return (unsigned char*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned char>(unsigned char) ;

template<> inline const unsigned short* NP::values<unsigned short>() const { return (unsigned short*)data.data() ; }
template<> inline       unsigned short* NP::values<unsigned short>()      {  return (unsigned short*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned short>(unsigned short) ;

template<> inline const unsigned int* NP::values<unsigned int>() const { return (unsigned int*)data.data() ; }
template<> inline       unsigned int* NP::values<unsigned int>()      {  return (unsigned int*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned int>(unsigned int) ;

template<> inline const unsigned long* NP::values<unsigned long>() const { return (unsigned long*)data.data() ; }
template<> inline       unsigned long* NP::values<unsigned long>()      {  return (unsigned long*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned long>(unsigned long) ;

template<> inline const unsigned long long* NP::values<unsigned long long>() const { return (unsigned long long*)data.data() ; }
template<> inline       unsigned long long* NP::values<unsigned long long>()      {  return (unsigned long long*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned long long>(unsigned long long) ;




template <typename T> void NP::Write(const char* dir, const char* name, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_ ) // static
{
    std::string dtype = descr_<T>::dtype() ; 

    std::cout 
        << "xNP::Write"
        << " dtype " << dtype
        << " ni  " << std::setw(7) << ni_
        << " nj  " << nj_
        << " nk  " << nk_
        << " nl  " << nl_
        << " nm  " << nm_
        << " dir " << std::setw(50) << dir
        << " name " << name
        << std::endl 
        ;   

    NP a(dtype.c_str(), ni_,nj_,nk_,nl_,nm_) ;    

    T* v = a.values<T>(); 

    int ni = std::max(1,ni_); 
    int nj = std::max(1,nj_); 
    int nk = std::max(1,nk_); 
    int nl = std::max(1,nl_); 
    int nm = std::max(1,nm_); 

    for(int i=0 ; i < ni ; i++ ) 
    for(int j=0 ; j < nj ; j++ )
    for(int k=0 ; k < nk ; k++ )
    for(int l=0 ; l < nl ; l++ )
    for(int m=0 ; m < nm ; m++ )
    {   
        int index = i*nj*nk*nl*nm + j*nk*nl*nm + k*nl*nm + l*nm + m  ;
        *(v + index) = *(data + index ) ; 
    }   
    a.save(dir, name); 
}


template void NP::Write<float>(   const char*, const char*, const float*,        int, int, int, int, int ); 
template void NP::Write<double>(  const char*, const char*, const double*,       int, int, int, int, int ); 
template void NP::Write<int>(     const char*, const char*, const int*,          int, int, int, int, int ); 
template void NP::Write<unsigned>(const char*, const char*, const unsigned*,     int, int, int, int, int ); 


template<typename T> void NP::Write(const char* dir, const char* name, const std::vector<T>& values )
{
    NP::Write(dir, name, values.data(), values.size() ); 
}

template void NP::Write<float>(   const char*, const char*, const std::vector<float>& ); 
template void NP::Write<double>(  const char*, const char*, const std::vector<double>&  ); 
template void NP::Write<int>(     const char*, const char*, const std::vector<int>& ); 
template void NP::Write<unsigned>(const char*, const char*, const std::vector<unsigned>& ); 

inline void NP::WriteNames(const char* dir, const char* name, const std::vector<std::string>& names, unsigned num_names_ )
{
    unsigned num_names = num_names_ == 0 ? names.size() : num_names_ ; 
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    std::ofstream stream(path.c_str(), std::ios::out|std::ios::binary);
    assert( num_names <= names.size() ); 
    for( unsigned i=0 ; i < num_names ; i++) stream << names[i] << std::endl ; 
    stream.close(); 
}



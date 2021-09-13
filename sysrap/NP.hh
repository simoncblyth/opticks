#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>
#include <cstdint>
#include <limits>
#include <random>

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
    union UIF32
    {
        std::uint32_t u ;
        std::int32_t  i ; 
        float         f ;  
    };            

    union UIF64
    {
        std::uint64_t  u ;
        std::int64_t   i ; 
        double         f ;  
    };            


    template<typename T> static NP*  Make( int ni_=-1, int nj_=-1, int nk_=-1, int nl_=-1, int nm_=-1 );  // dtype from template type
    template<typename T> static NP*  Linspace( T x0, T x1, unsigned nx ); 
    template<typename T> static NP*  MakeUniform( unsigned ni, unsigned seed=0u );  
    template<typename T> static unsigned NumSteps( T x0, T x1, T dx ); 

    NP(const char* dtype_="<f4", int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    void init(); 
    void set_shape( int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1); 
    void set_shape( const std::vector<int>& src_shape ); 
    bool has_shape(int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ) const ;  

    void set_dtype(const char* dtype_); // *set_dtype* may change shape and size of array while retaining the same underlying bytes 

    static void sizeof_check(); 


    template<typename T> static int DumpCompare( const NP* a, const NP* b, unsigned a_column, unsigned b_column, const T epsilon ); 

    static int Memcmp( const NP* a, const NP* b ); 
    static NP* Concatenate(const std::vector<NP*>& aa); 
    static NP* Concatenate(const char* dir, const std::vector<std::string>& names); 

    static NP* Combine(const std::vector<const NP*>& aa, bool annotate=true); 

    // load array asis 
    static NP* Load(const char* path); 
    static NP* Load(const char* dir, const char* name); 
    static NP* Load(const char* dir, const char* reldir, const char* name); 

    // load float OR double array and if float(4 bytes per element) widens it to double(8 bytes per element)  
    static NP* LoadWide(const char* path); 
    static NP* LoadWide(const char* dir, const char* name); 
    static NP* LoadWide(const char* dir, const char* reldir, const char* name); 

    // load float OR double array and if double(8 bytes per element) narrows it to float(4 bytes per element)  
    static NP* LoadNarrow(const char* path); 
    static NP* LoadNarrow(const char* dir, const char* name); 
    static NP* LoadNarrow(const char* dir, const char* reldir, const char* name); 


    static NP* MakeDemo(const char* dtype="<f4" , int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 

    template<typename T> static void Write(const char* dir, const char* name, const std::vector<T>& values ); 
    template<typename T> static void Write(const char* dir, const char* name, const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    template<typename T> static void Write(const char* dir, const char* reldir, const char* name, const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 

    template<typename T> static void Write(const char* path                 , const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 

    static void WriteNames(const char* dir, const char* name, const std::vector<std::string>& names, unsigned num_names=0 ); 
    static void WriteNames(const char* dir, const char* reldir, const char* name, const std::vector<std::string>& names, unsigned num_names=0 ); 
    static void WriteNames(const char* path,                  const std::vector<std::string>& names, unsigned num_names=0 ); 

    static void ReadNames(const char* dir, const char* name, std::vector<std::string>& names ) ;
    static void ReadNames(const char* path,                  std::vector<std::string>& names ) ;


    template<typename T> T*       values() ; 
    template<typename T> const T*  cvalues() const  ; 

    template<typename T> void fill(T value); 
    template<typename T> void _fillIndexFlat(T offset=0); 
    template<typename T> void _dump(int i0=-1, int i1=-1) const ;   


    static NP* MakeLike(  const NP* src);  
    static NP* MakeNarrow(const NP* src); 
    static NP* MakeWide(  const NP* src); 
    static NP* MakeCopy(  const NP* src); 
    NP* copy() const ; 

    bool is_pshaped() const ; 
    template<typename T> T    plhs(unsigned column ) const ;  
    template<typename T> T    prhs(unsigned column ) const ;  
    template<typename T> int  pfindbin(const T value, unsigned column, bool& in_range ) const ;  
    template<typename T> void get_edges(T& lo, T& hi, unsigned column, int ibin) const ; 

    template<typename T> T    pdomain(const T value, int item=-1, bool dump=false  ) const ; 

    template<typename T> T    psum(unsigned column ) const ;  
    template<typename T> void pscale(T scale, unsigned column);
    template<typename T> void pscale_add(T scale, T add, unsigned column);
    template<typename T> void pdump(const char* msg="NP::pdump") const ; 
    template<typename T> void minmax(T& mn, T&mx, unsigned j=1 ) const ; 
    template<typename T> void linear_crossings( T value, std::vector<T>& crossings ) const ; 
    template<typename T> T    trapz() const ;                      // composite trapezoidal integration, requires pshaped
    template<typename T> T    interp(T x) const ;                  // requires pshaped 
    template<typename T> T    interp(unsigned iprop, T x) const ;  // requires NP::Combine of pshaped arrays 
    template<typename T> NP*  cumsum(int axis=0) const ; 
    template<typename T> void divide_by_last() ; 



    template<typename T> void read(const T* data);
    template<typename T> void read2(const T* data);

    template<typename T> std::string _present(T v) const ; 

    void fillIndexFlat(); 
    void dump(int i0=-1, int i1=-1) const ; 

    void clear();   

    static bool Exists(const char* dir, const char* name);   
    static bool Exists(const char* path);   
    int load(const char* dir, const char* name);   
    int load(const char* path);   
    int load_meta( const char* path ); 


    static std::string form_name(const char* stem, const char* ext); 
    static std::string form_path(const char* dir, const char* name);   
    static std::string form_path(const char* dir, const char* reldir, const char* name);   

    void save_header(const char* path);   
    void old_save(const char* path) ;  // formerly the *save* methods could not be const because of update_headers
    void save(const char* path) const ;  // *save* methods now can be const due to dynamic creation of header
    void save(const char* dir, const char* name) const ;   
    void save(const char* dir, const char* reldir, const char* name) const ;   

    std::string get_jsonhdr_path() const ; // .npy -> .npj on loaded path
    void save_jsonhdr() const ;    
    void save_jsonhdr(const char* path) const ;   
    void save_jsonhdr(const char* dir, const char* name) const ;   

    std::string desc() const ; 
    void set_meta( const std::vector<std::string>& lines, char delim='\n' ); 
    void get_meta( std::vector<std::string>& lines,       char delim='\n' ) const ; 

    char*       bytes();  
    const char* bytes() const ;  

    unsigned num_values() const ; 
    unsigned num_itemvalues() const ; 
    unsigned arr_bytes() const ;   // formerly num_bytes
    unsigned item_bytes() const ;   // *item* comprises all dimensions beyond the first 
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

HMM : do not like this as it prevents NP::save from being const 

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
inline unsigned NP::num_itemvalues() const { return NPS::itemsize(shape) ;  }
inline unsigned NP::arr_bytes()  const { return NPS::size(shape)*ebyte ; }
inline unsigned NP::item_bytes() const { return NPS::itemsize(shape)*ebyte ; }
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
    init(); 
}

inline void NP::init()
{
    data.resize( size*ebyte ) ;  // vector of char  
    std::fill( data.begin(), data.end(), 0 );     
    _prefix.assign(net_hdr::LENGTH, '\0' ); 
    _hdr = make_header(); 
}

inline void NP::set_shape(int ni, int nj, int nk, int nl, int nm)
{
    size = NPS::copy_shape(shape, ni, nj, nk, nl, nm); 
    init(); 
}
inline void NP::set_shape(const std::vector<int>& src_shape)
{
    size = NPS::copy_shape(shape, src_shape); 
    init(); 
}

inline bool NP::has_shape(int ni, int nj, int nk, int nl, int nm) const 
{
    unsigned ndim = shape.size() ; 
    return 
           ( ni == -1 || ( ndim > 0 && int(shape[0]) == ni)) && 
           ( nj == -1 || ( ndim > 1 && int(shape[1]) == nj)) && 
           ( nk == -1 || ( ndim > 2 && int(shape[2]) == nk)) && 
           ( nl == -1 || ( ndim > 3 && int(shape[3]) == nl)) && 
           ( nm == -1 || ( ndim > 4 && int(shape[4]) == nm))  
           ;
}












template<typename T> inline const T*  NP::cvalues() const { return (T*)data.data() ;  } 
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



/**

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
specialize(){ specialize-types | while read t ; do specialize- "$t" ; done  ; }
specialize

**/

// template specializations generated by above bash function

template<>  inline const float* NP::cvalues<float>() const { return (float*)data.data() ; }
template<>  inline       float* NP::values<float>()      {  return (float*)data.data() ; }
template    void NP::_fillIndexFlat<float>(float) ;

template<> inline const double* NP::cvalues<double>() const { return (double*)data.data() ; }
template<> inline       double* NP::values<double>()      {  return (double*)data.data() ; }
template   void NP::_fillIndexFlat<double>(double) ;

template<> inline const char* NP::cvalues<char>() const { return (char*)data.data() ; }
template<> inline       char* NP::values<char>()      {  return (char*)data.data() ; }
template   void NP::_fillIndexFlat<char>(char) ;

template<> inline const short* NP::cvalues<short>() const { return (short*)data.data() ; }
template<> inline       short* NP::values<short>()      {  return (short*)data.data() ; }
template   void NP::_fillIndexFlat<short>(short) ;

template<> inline const int* NP::cvalues<int>() const { return (int*)data.data() ; }
template<> inline       int* NP::values<int>()      {  return (int*)data.data() ; }
template   void NP::_fillIndexFlat<int>(int) ;

template<> inline const long* NP::cvalues<long>() const { return (long*)data.data() ; }
template<> inline       long* NP::values<long>()      {  return (long*)data.data() ; }
template   void NP::_fillIndexFlat<long>(long) ;

template<> inline const long long* NP::cvalues<long long>() const { return (long long*)data.data() ; }
template<> inline       long long* NP::values<long long>()      {  return (long long*)data.data() ; }
template   void NP::_fillIndexFlat<long long>(long long) ;

template<> inline const unsigned char* NP::cvalues<unsigned char>() const { return (unsigned char*)data.data() ; }
template<> inline       unsigned char* NP::values<unsigned char>()      {  return (unsigned char*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned char>(unsigned char) ;

template<> inline const unsigned short* NP::cvalues<unsigned short>() const { return (unsigned short*)data.data() ; }
template<> inline       unsigned short* NP::values<unsigned short>()      {  return (unsigned short*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned short>(unsigned short) ;

template<> inline const unsigned int* NP::cvalues<unsigned int>() const { return (unsigned int*)data.data() ; }
template<> inline       unsigned int* NP::values<unsigned int>()      {  return (unsigned int*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned int>(unsigned int) ;

template<> inline const unsigned long* NP::cvalues<unsigned long>() const { return (unsigned long*)data.data() ; }
template<> inline       unsigned long* NP::values<unsigned long>()      {  return (unsigned long*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned long>(unsigned long) ;

template<> inline const unsigned long long* NP::cvalues<unsigned long long>() const { return (unsigned long long*)data.data() ; }
template<> inline       unsigned long long* NP::values<unsigned long long>()      {  return (unsigned long long*)data.data() ; }
template   void NP::_fillIndexFlat<unsigned long long>(unsigned long long) ;


/**
NP::MakeLike
--------------

Creates an array of the same shape and type as the *src* array.
Values are *NOT* copied from *src*. 

**/

inline NP* NP::MakeLike(const NP* src) // static 
{
    NP* dst = new NP(src->dtype); 
    dst->set_shape(src->shape) ; 
    return dst ; 
}

inline NP* NP::MakeNarrow(const NP* a) // static 
{
    assert( a->ebyte == 8 ); 
    std::string b_dtype = NPU::_make_narrow(a->dtype); 

    NP* b = new NP(b_dtype.c_str()); 
    b->set_shape( a->shape ); 

    assert( a->num_values() == b->num_values() ); 
    unsigned nv = a->num_values(); 

    if( a->uifc == 'f' && b->uifc == 'f')
    {
        const double* aa = a->cvalues<double>() ;  
        float* bb = b->values<float>() ;  
        for(unsigned i=0 ; i < nv ; i++)
        {
            bb[i] = float(aa[i]); 
        }
    }

    std::cout 
        << "NP::MakeNarrow"
        << " a.dtype " << a->dtype
        << " b.dtype " << b->dtype
        << std::endl 
        ;

    return b ; 
}


inline NP* NP::MakeWide(const NP* a) // static 
{
    assert( a->ebyte == 4 ); 
    std::string b_dtype = NPU::_make_wide(a->dtype); 

    NP* b = new NP(b_dtype.c_str()); 
    b->set_shape( a->shape ); 

    assert( a->num_values() == b->num_values() ); 
    unsigned nv = a->num_values(); 

    if( a->uifc == 'f' && b->uifc == 'f')
    {
        const float* aa = a->cvalues<float>() ;  
        double* bb = b->values<double>() ;  
        for(unsigned i=0 ; i < nv ; i++)
        {
            bb[i] = double(aa[i]); 
        }
    }

    std::cout 
        << "NP::MakeWide"
        << " a.dtype " << a->dtype
        << " b.dtype " << b->dtype
        << std::endl 
        ;

    return b ; 
}

inline NP* NP::MakeCopy(const NP* a) // static 
{
    NP* b = new NP(a->dtype); 
    b->set_shape( a->shape ); 
    assert( a->arr_bytes() == b->arr_bytes() ); 

    memcpy( b->bytes(), a->bytes(), a->arr_bytes() );    
    unsigned nv = a->num_values(); 

    std::cout 
        << "NP::MakeCopy"
        << " a.dtype " << a->dtype
        << " b.dtype " << b->dtype
        << " nv " << nv
        << std::endl 
        ;

    return b ; 
}

inline NP* NP::copy() const 
{
    return MakeCopy(this); 
}


inline NP* NP::LoadWide(const char* dir, const char* reldir, const char* name)
{
    std::string path = form_path(dir, reldir, name); 
    return LoadWide(path.c_str());
}

inline NP* NP::LoadWide(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    return LoadWide(path.c_str());
}

/**
NP::LoadWide
--------------

Loads array and widens it to 8 bytes per element if not already wide.

**/
inline NP* NP::LoadWide(const char* path)
{
    NP* a = NP::Load(path);  

    assert( a->uifc == 'f' && ( a->ebyte == 8 || a->ebyte == 4 ));  
    // cannot think of application for doing this with  ints, so restrict to float OR double 

    NP* b = a->ebyte == 8 ? NP::MakeCopy(a) : NP::MakeWide(a) ;  

    a->clear(); 

    return b ; 
}


inline NP* NP::LoadNarrow(const char* dir, const char* reldir, const char* name)
{
    std::string path = form_path(dir, reldir, name); 
    return LoadNarrow(path.c_str());
}

inline NP* NP::LoadNarrow(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    return LoadNarrow(path.c_str());
}


/**
NP::LoadNarrow
---------------

Loads array and narrows to 4 bytes per element if not already narrow.

**/
inline NP* NP::LoadNarrow(const char* path)
{
    NP* a = NP::Load(path);  

    assert( a->uifc == 'f' && ( a->ebyte == 8 || a->ebyte == 4 ));  
    // cannot think of application for doing this with  ints, so restrict to float OR double 

    NP* b = a->ebyte == 4 ? NP::MakeCopy(a) : NP::MakeNarrow(a) ;  

    a->clear(); 

    return b ; 
}









inline bool NP::is_pshaped() const
{
    bool property_shaped = shape.size() == 2 && shape[1] == 2 && shape[0] > 1 ;
    return property_shaped ;  
}


template<typename T> inline T NP::plhs(unsigned column) const 
{
    const T* vv = cvalues<T>(); 

    unsigned ndim = shape.size() ; 
    assert( ndim == 1 || ndim == 2); 

    unsigned nj = ndim == 1 ? 1 : shape[1] ; 
    assert( column < nj ); 
 
    const T lhs = vv[nj*(0)+column] ; 
    return lhs ;  
}


template<typename T> inline T NP::prhs(unsigned column) const 
{
    const T* vv = cvalues<T>(); 

    unsigned ndim = shape.size() ; 
    unsigned ni = shape[0] ;
    unsigned nj = ndim == 1 ? 1 : shape[1] ; 
    assert( column < nj ); 

    const T rhs = vv[nj*(ni-1)+column] ; 

#ifdef DEBUG
    /*
    std::cout 
        << "NP::prhs"
        << " column " << std::setw(3) << column 
        << " ndim " << std::setw(3) << ndim 
        << " ni " << std::setw(3) << ni 
        << " nj " << std::setw(3) << nj 
        << " rhs " << std::setw(10) << std::fixed << std::setprecision(4) << rhs 
        << std::endl 
        ;
     */
#endif

    return rhs ;  
}



/**
NP::pfindbin
---------------

Return *ibin* index of bin corresponding to the argument value.

+---------------------+------------------+----------------------+ 
|  condition          |   ibin           |  in_range            |
+=====================+==================+======================+
|  value < lhs        |   0              |   false              | 
+---------------------+------------------+----------------------+ 
|  value == lhs       |   1              |   true               | 
+---------------------+------------------+----------------------+ 
|  lhs < value < rhs  |   1 .. ni-1      |   true               |
+---------------------+------------------+----------------------+ 
|  value == rhs       |   ni             |   false              | 
+---------------------+------------------+----------------------+ 
|  value > rhs        |   ni             |   false              | 
+---------------------+------------------+----------------------+ 

Example indices for bins array of shape (4,) with 3 bins and 4 values (ni=4)
This numbering scheme matches that used by np.digitize::

        
                +-------------+--------------+-------------+         
                |             |              |             |
                |             |              |             |
                +-------------+--------------+-------------+         
              0        1             2               3            4 

                lhs                                       rhs

**/

template<typename T> inline int  NP::pfindbin(const T value, unsigned column, bool& in_range) const 
{
    const T* vv = cvalues<T>(); 

    unsigned ndim = shape.size() ; 
    unsigned ni = shape[0] ;
    unsigned nj = ndim == 1 ? 1 : shape[1] ; 
    assert( column < nj ); 
 
    const T lhs = vv[nj*(0)+column] ; 
    const T rhs = vv[nj*(ni-1)+column] ; 
   
    int ibin = -1 ; 
    in_range = false ; 
    if( value < lhs )         // value==lhs is in_range 
    {
        ibin = 0 ; 
    }
    else if( value >= rhs )   // value==rhs is NOT in_range 
    {
        ibin = ni ; 
    }
    else if ( value >= lhs && value < rhs )
    {
        in_range = true ; 
        for(unsigned i=0 ; i < ni-1 ; i++) 
        {
            const T v0 = vv[nj*(i+0)+column] ; 
            const T v1 = vv[nj*(i+1)+column] ; 
            if( value >= v0 && value < v1 )
            {
                 ibin = i + 1 ;  // maximum i from here is: ni-1-1 -> max ibin is ni-1
                 break ; 
            } 
        }
    } 
    return ibin ; 
}
 




/**
NP::get_edges
----------------

Return bin edges using numbering convention from NP::pfindbin, 
for out of range ibin == 0 returns lhs edge for both lo and hi
for out of range ibin = ni returns rhs edge for both lo and hi. 

**/

template<typename T> inline void  NP::get_edges(T& lo, T& hi, unsigned column, int ibin) const 
{
    const T* vv = cvalues<T>(); 

    unsigned ndim = shape.size() ; 
    unsigned ni = shape[0] ;
    unsigned nj = ndim == 1 ? 1 : shape[1] ; 
    assert( column < nj ); 
 
    const T lhs = vv[nj*(0)+column] ; 
    const T rhs = vv[nj*(ni-1)+column] ; 

    if( ibin == 0 )
    {
        lo = lhs ; 
        hi = lhs ; 
    }   
    else if( ibin == ni )
    {
        lo = rhs ; 
        hi = rhs ; 
    }   
    else
    {
        unsigned i = ibin - 1 ; 
        lo  = vv[nj*(i)+column] ; 
        hi  = vv[nj*(i+1)+column] ; 
    }
}



/**
NP::pdomain
-------------

Returns the domain (eg energy or wavelength) corresponding 
to the property value argument. 

Requires arrays of shape (num_dom, 2) when item is at default value of -1 

Also support arrys of shape (num_item, num_dom, 2 ) when item is used to pick the item. 


   
                                                        1  (x1,y1)     (  binVector[bin+1], dataVector[bin+1] )
                                                       /
                                                      /
                                                     *  ( xv,yv )       ( res, aValue )      
                                                    /
                                                   /
                                                  0  (x0,y0)          (  binVector[bin], dataVector[bin] )


              Similar triangles::
               
                 xv - x0       x1 - x0 
               ---------- =   -----------
                 yv - y0       y1 - y0 

                                                   x1 - x0
                   xv  =    x0  +   (yv - y0) *  -------------- 
                                                   y1 - y0

**/

template<typename T> inline T  NP::pdomain(const T value, int item, bool dump ) const 
{
    const T zero = 0. ; 
    unsigned ndim = shape.size() ; 
    assert( ndim == 2 || ndim == 3 ); 
    unsigned ni = shape[ndim-2]; 
    unsigned nj = shape[ndim-1]; 
    assert( nj == 2 ); 

    unsigned num_items = ndim == 3 ? shape[0] : 1 ; 
    assert( item < int(num_items) ); 
    unsigned item_offset = item == -1 ? 0 : ni*nj*item ; 

    const T* vv = cvalues<T>() + item_offset ;  // shortcut approach to handling multiple items 

    enum { DOM=0 , VAL=1 } ; 

    const T lhs_dom = vv[nj*(0)+DOM]; 
    const T rhs_dom = vv[nj*(ni-1)+DOM];
    assert( rhs_dom > lhs_dom ); 

    const T lhs_val = vv[nj*(0)+VAL]; 
    const T rhs_val = vv[nj*(ni-1)+VAL];
    assert( rhs_val > lhs_val ); 


    const T yv = value ; 
    T xv ;   

    if( yv <= lhs_val )
    {
        xv = lhs_dom ; 
    }
    else if( yv >= rhs_val )
    {
        xv = rhs_dom  ; 
    }
    else if ( yv >= lhs_val && yv < rhs_val  )
    {
        for(unsigned i=0 ; i < ni-1 ; i++) 
        {
            const T x0 = vv[nj*(i+0)+DOM] ; 
            const T y0 = vv[nj*(i+0)+VAL] ; 
            const T x1 = vv[nj*(i+1)+DOM] ; 
            const T y1 = vv[nj*(i+1)+VAL] ;

            const T dy = y1 - y0 ;  
            assert( dy >= zero );   // must be monotonic for this to make sense

            if( y0 <= yv && yv < y1 )
            { 
                xv = x0 ; 
                if( dy > zero ) xv += (yv-y0)*(x1-x0)/dy ; 
                break ;   
            }
        }
    } 

    if(dump)
    {
        std::cout 
            << "NP::pdomain.dump "
            << " item " << std::setw(4) << item
            << " ni " << std::setw(4) << ni
            << " lhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_dom
            << " rhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_dom
            << " lhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_val
            << " rhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_val
            << " yv " << std::setw(10) << std::fixed << std::setprecision(4) << yv
            << " xv " << std::setw(10) << std::fixed << std::setprecision(4) << xv
            << std::endl 
            ; 
    }


#ifdef DEBUG
    std::cout 
        << "NP::pdomain"
        << " item " << std::setw(4) << item
        << " ni " << std::setw(4) << ni
        << " yv " << std::setw(10) << std::fixed << std::setprecision(4) << yv
        << " xv " << std::setw(10) << std::fixed << std::setprecision(4) << xv
        << " lhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_dom
        << " rhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_dom
        << " lhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_val
        << " rhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_val
        << std::endl 
        ;
#endif
    return xv ; 
}


template<typename T> inline T  NP::psum(unsigned column) const 
{
    const T* vv = cvalues<T>(); 
    unsigned ni = shape[0] ;
    unsigned ndim = shape.size() ; 
    unsigned nj = ndim == 1 ? 1 : shape[1] ; 
    assert( column < nj ); 

    T sum = 0. ; 
    for(unsigned i=0 ; i < ni ; i++) sum += vv[nj*i+column] ;  
    return sum ; 
}



template<typename T> inline void NP::pscale_add(T scale, T add, unsigned column)
{
    assert( is_pshaped() ); 
    assert( column < 2 ); 
    T* vv = values<T>(); 
    unsigned ni = shape[0] ; 
    for(unsigned i=0 ; i < ni ; i++) vv[2*i+column] = vv[2*i+column]*scale + add ;  ; 
}

template<typename T> inline void NP::pscale(T scale, unsigned column)
{
    pscale_add(scale, T(0.), column ); 
}



template<typename T> inline void NP::pdump(const char* msg) const  
{
    bool property_shaped = is_pshaped(); 
    assert( property_shaped ); 

    unsigned ni = shape[0] ; 
    std::cout << msg << " ni " << ni << std::endl ; 

    const T* vv = cvalues<T>(); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        std::cout 
             << " i " << std::setw(3) << i 
             << " px " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+0] 
             << " py " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+1] 
             << std::endl
             ; 
    }
}

template<typename T> inline void NP::minmax(T& mn, T&mx, unsigned j ) const 
{
    unsigned ndim = shape.size() ; 

    assert( ndim == 2 );   // TODO: support ndim 3 with item argument 
    unsigned ni = shape[0] ; 
    unsigned nj = shape[1] ; 
    assert( j < nj ); 

    mn = std::numeric_limits<T>::max() ; 
    mx = std::numeric_limits<T>::min() ; 

    const T* vv = cvalues<T>(); 
    for(unsigned i=0 ; i < ni ; i++)
    {
        T v = vv[nj*i+j] ; 
        if( v > mx ) mx = v; 
        if( v < mn ) mn = v; 
    }
}




/**
NP::linear_crossings
------------------------

As linearly interpolated properties eg RINDEX using NP::interp 
are piecewise linear functions it is possible to find the 
crossings between such functions and constant values 
without using optimization. Simply observing sign changes to identify 
crossing bins and then some linear calc provides the roots::

             
                  (x1,v1)
                   *
                  / 
                 /
                /
        -------?(x,v)----    v = 0    when values are "ri_ - BetaInverse"
              /
             /
            /
           /
          * 
        (x0,v0)      


         Only x is unknown 


              v1 - v        v - v0
             ----------  =  ----------
              x1 - x        x - x0  


           v1 (x - x0 ) =  -v0  (x1 - x )

           v1.x - v1.x0 = - v0.x1 + v0.x  

           ( v1 - v0 ) x = v1*x0 - v0*x1


                         v1*x0 - v0*x1
               x    =   -----------------
                          ( v1 - v0 ) 


Developed in opticks/ana/rindex.py for an attempt to developing inverse transform Cerenkov RINDEX 
sampling.

**/

template<typename T> inline void NP::linear_crossings( T value, std::vector<T>& crossings ) const 
{
    assert( shape.size() == 2 && shape[1] == 2 && shape[0] > 1); 
    unsigned ni = shape[0] ; 
    const T* vv = cvalues<T>(); 
    crossings.clear(); 

    for(unsigned i=0 ; i < ni-1 ; i++ )
    { 
        T x0 = vv[2*(i+0)+0] ; 
        T x1 = vv[2*(i+1)+0] ; 
        T v0 = value - vv[2*(i+0)+1] ; 
        T v1 = value - vv[2*(i+1)+1] ; 
        if( v0*v1 < T(0.))
        {
            T x = (v1*x0 - v0*x1)/(v1-v0) ; 
            //printf("i %d x0 %6.4f x1 %6.4f v0 %6.4f v1 %6.4f x %6.4f \n", i, x0,x1,v0,v1,x) ; 
            crossings.push_back(x) ; 
        }
    }
} 

/**
NP::trapz
-----------

Composite trapezoidal numerical integration

* https://en.wikipedia.org/wiki/Trapezoidal_rule

**/

template<typename T> inline T NP::trapz() const 
{
    assert( shape.size() == 2 && shape[1] == 2 && shape[0] > 1); 
    unsigned ni = shape[0] ; 
    const T* vv = cvalues<T>(); 

    T half(0.5); 
    T integral = T(0.);  
    for(unsigned i=0 ; i < ni-1 ; i++)
    {
        unsigned i0 = i+0 ; 
        unsigned i1 = i+1 ; 

        T x0 = vv[2*i0+0] ; 
        T y0 = vv[2*i0+1] ; 

        T x1 = vv[2*i1+0] ; 
        T y1 = vv[2*i1+1] ; 

        integral += (x1 - x0)*(y0 + y1)*half ;  
    } 
    return integral ;  
}



/**
NP::interp
------------

CAUTION: using the wrong type here somehow scrambles the array contents, 
so always explicitly define the template type : DO NOT RELY ON COMPILER WORKING IT OUT.

**/

template<typename T> inline T NP::interp(T x) const  
{
    assert( shape.size() == 2 && shape[1] == 2 && shape[0] > 1); 
    unsigned ni = shape[0] ; 

    //pdump<T>("NP::interp.pdump (not being explicit with the type managed to scramble array content) ");  

    const T* vv = cvalues<T>(); 

    int lo = 0 ;
    int hi = ni-1 ;

/*
    std::cout 
         << " NP::interp "
         << " x " << x 
         << " ni " << ni 
         << " lo " << lo
         << " hi " << hi
         << " vx_lo " << vv[2*lo+0] 
         << " vy_lo " <<  vv[2*lo+1] 
         << " vx_hi " << vv[2*hi+0] 
         << " vy_hi " <<  vv[2*hi+1] 
         << std::endl
         ; 
*/

    if( x <= vv[2*lo+0] ) return vv[2*lo+1] ; 
    if( x >= vv[2*hi+0] ) return vv[2*hi+1] ; 

    while (lo < hi-1)
    {
        int mi = (lo+hi)/2;
        if (x < vv[2*mi+0]) hi = mi ;
        else lo = mi;
    }

    T dy = vv[2*hi+1] - vv[2*lo+1] ; 
    T dx = vv[2*hi+0] - vv[2*lo+0] ; 
    T y = vv[2*lo+1] + dy*(x-vv[2*lo+0])/dx ; 
    return y ; 
}






template<typename T> inline T NP::interp(unsigned iprop, T x) const  
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 3 && shape[ndim-1] == 2 && iprop < shape[0] && shape[1] > 1 ); 

    unsigned niv = num_itemvalues() ; 
    const T* vv = cvalues<T>() + iprop*niv ; 

    unsigned ni(0) ; 
    if( ebyte == 4 )
    {
        UIF32 uif32 ; 
        uif32.f = *( vv + niv - 1) ; 
        ni = uif32.u ; 
    }
    else if( ebyte == 8 )
    {
        UIF64 uif64 ; 
        uif64.f = *( vv + niv - 1) ; 
        ni = uif64.u ;   // narrowing doesnt matter, as unsigned will be big enough 
    }

    int lo = 0 ;
    int hi = ni-1 ;

/*
    std::cout 
         << " NP::interp "
         << " x " << x 
         << " ni " << ni 
         << " lo " << lo
         << " hi " << hi
         << " vx_lo " << vv[2*lo+0] 
         << " vy_lo " <<  vv[2*lo+1] 
         << " vx_hi " << vv[2*hi+0] 
         << " vy_hi " <<  vv[2*hi+1] 
         << std::endl
         ; 
*/

    if( x <= vv[2*lo+0] ) return vv[2*lo+1] ; 
    if( x >= vv[2*hi+0] ) return vv[2*hi+1] ; 

    while (lo < hi-1)
    {
        int mi = (lo+hi)/2;
        if (x < vv[2*mi+0]) hi = mi ;
        else lo = mi;
    }

    T dy = vv[2*hi+1] - vv[2*lo+1] ; 
    T dx = vv[2*hi+0] - vv[2*lo+0] ; 
    T y = vv[2*lo+1] + dy*(x-vv[2*lo+0])/dx ; 
    return y ; 
}

template<typename T> inline NP* NP::cumsum(int axis) const  
{
    assert( axis == 1 && "for now only axis=1 implemented" ); 
    const T* vv = cvalues<T>(); 
    NP* cs = NP::MakeLike(this) ; 
    T* ss = cs->values<T>(); 
    for(unsigned p=0 ; p < size ; p++) ss[p] = vv[p] ;   // flat copy 

    unsigned ndim = shape.size() ; 

    if( ndim == 1 )
    {
        unsigned ni = shape[0] ; 
        for(unsigned i=1 ; i < ni ; i++) ss[i] += ss[i-1] ;  
    }
    else if( ndim == 2 )
    {
        unsigned ni = shape[0] ; 
        unsigned nj = shape[1] ; 
        for(unsigned i=0 ; i < ni ; i++)
        { 
            for(unsigned j=1 ; j < nj ; j++) ss[i*nj+j] += ss[i*nj+j-1] ;  
        }
    }
    else
    {
        assert( 0 && "for now only 1d or 2d implemented");  
    }
    return cs ; 
}


/**
NP::divide_by_last
--------------------


**/

template<typename T> inline void NP::divide_by_last() 
{
    unsigned ndim = shape.size() ; 
    T* vv = values<T>(); 
    const T zero(0.); 

    if( ndim == 1 )
    {
        unsigned ni = shape[0] ; 
        for(unsigned i=0 ; i < ni ; i++) vv[i] = vv[i]/vv[ni-1] ;  
    }
    else if( ndim == 2 )
    {
        unsigned ni = shape[0] ; 
        unsigned nj = shape[1] ; 
        for(unsigned i=0 ; i < ni ; i++)
        {
            const T last = vv[i*nj+nj-1] ;  
            for(unsigned j=0 ; j < nj ; j++) if(last != zero) vv[i*nj+j] /= last ;  
        }
    }
    else if( ndim == 3 )   // eg (1000, 100, 2)    1000(diff BetaInverse) * 100 * (energy, integral)  
    {
        unsigned ni = shape[0] ; 
        unsigned nj = shape[1] ; 
        unsigned nk = shape[2] ; 
        assert( nk == 2 ); // not required by below, but for sanity of understanding 

        for(unsigned i=0 ; i < ni ; i++)
        {
            unsigned k = nk - 1 ;                         // eg the property, not the domain energy in k=0
            const T last = vv[i*nj*nk+(nj-1)*nk+k] ;  
            for(unsigned j=0 ; j < nj ; j++) if(last != zero) vv[i*nj*nk+j*nk+k] /= last ;  
        }
    }
    else
    {
        assert( 0 && "for now only ndim 1,2,3 implemented");  
    }
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

inline void NP::set_meta( const std::vector<std::string>& lines, char delim )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < lines.size() ; i++) ss << lines[i] << delim  ; 
    meta = ss.str(); 
}

inline void NP::get_meta( std::vector<std::string>& lines, char delim  ) const 
{
    if(meta.empty()) return ; 

    std::stringstream ss ; 
    ss.str(meta.c_str())  ;
    std::string s;
    while (std::getline(ss, s, delim)) lines.push_back(s) ; 
}

template<typename T>
inline int NP::DumpCompare( const NP* a, const NP* b , unsigned a_column, unsigned b_column, const T epsilon ) // static
{
    const T* aa = a->cvalues<T>(); 
    const T* bb = b->cvalues<T>(); 

    unsigned a_ndim = a->shape.size() ; 
    unsigned a_ni = a->shape[0] ;
    unsigned a_nj = a_ndim == 1 ? 1 : a->shape[1] ; 
    assert( a_column < a_nj ); 
 
    unsigned b_ndim = b->shape.size() ; 
    unsigned b_ni = b->shape[0] ;
    unsigned b_nj = b_ndim == 1 ? 1 : b->shape[1] ; 
    assert( b_column < b_nj ); 
 
    assert( a_ni == b_ni ); 

    T av_sum = 0. ; 
    T bv_sum = 0. ; 
    int mismatch = 0 ; 

    for(unsigned i=0 ; i < a_ni ; i++)
    {
        const T av = aa[a_nj*i+a_column] ; 
        const T bv = bb[b_nj*i+b_column] ; 
        av_sum += av ; 
        bv_sum += bv ; 
        std::cout 
            << std::setw(4) << i 
            << " a " << std::setw(10) << std::fixed << std::setprecision(4) << av 
            << " b " << std::setw(10) << std::fixed << std::setprecision(4) << bv 
            << " a-b " << std::setw(10) << std::fixed << std::setprecision(4) << av-bv
            << std::endl 
            ;
        if(std::abs(av-bv) > epsilon) mismatch += 1 ;  
    }
    std::cout 
        << std::setw(4) << "sum" 
        << " a " << std::setw(10) << std::fixed << std::setprecision(4) << av_sum 
        << " b " << std::setw(10) << std::fixed << std::setprecision(4) << bv_sum 
        << " a-b " << std::setw(10) << std::fixed << std::setprecision(4) << av_sum-bv_sum
        << " mismatch " << mismatch
        << std::endl 
        ;
    return mismatch ; 
}


inline int NP::Memcmp(const NP* a, const NP* b ) // static
{
    unsigned a_bytes = a->arr_bytes() ; 
    unsigned b_bytes = b->arr_bytes() ; 
    return a_bytes == b_bytes ? memcmp(a->bytes(), b->bytes(), a_bytes) : -1 ; 
}

inline NP* NP::Concatenate(const char* dir, const std::vector<std::string>& names) // static 
{
    std::vector<NP*> aa ;
    for(unsigned i=0 ; i < names.size() ; i++)
    {
         const char* name = names[i].c_str(); 
         NP* a = Load(dir, name); 
         aa.push_back(a); 
    }
    NP* concat = NP::Concatenate(aa); 
    return concat ; 
}

inline NP* NP::Concatenate(const std::vector<NP*>& aa)  // static 
{
    assert( aa.size() > 0 ); 

    NP* a0 = aa[0] ; 
    
    unsigned nv0 = a0->num_itemvalues() ; 
    const char* dtype0 = a0->dtype ; 

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        NP* a = aa[i] ;

        unsigned nv = a->num_itemvalues() ; 
        bool compatible = nv == nv0 && strcmp(dtype0, a->dtype) == 0 ; 
        if(!compatible) 
            std::cout 
                << "NP::Concatenate FATAL expecting equal itemsize"
                << " nv " << nv 
                << " nv0 " << nv0 
                << " a.dtype " << a->dtype 
                << " dtype0 " << dtype0 
                << std::endl
                ; 
        assert(compatible);  

        std::cout << std::setw(3) << i << " " << a->desc() << " nv " << nv << std::endl ; 
    }

    unsigned ni_total = 0 ; 
    for(unsigned i=0 ; i < aa.size() ; i++) ni_total += aa[i]->shape[0] ; 
    std::cout << " ni_total " << ni_total << std::endl ; 

    std::vector<int> comb_shape ; 
    NPS::copy_shape( comb_shape, a0->shape );  
    comb_shape[0] = ni_total ; 

    NP* c = new NP(a0->dtype); 
    c->set_shape(comb_shape); 
    std::cout << " c " << c->desc() << std::endl ; 

    unsigned offset_bytes = 0 ; 
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        NP* a = aa[i]; 
        unsigned a_bytes = a->arr_bytes() ; 
        memcpy( c->data.data() + offset_bytes ,  a->data.data(),  a_bytes ); 
        offset_bytes += a_bytes ;  
        a->clear(); // HUH: THATS A BIT IMPOLITE ASSUMING CALLER DOESNT WANT TO USE INPUTS
    }
    return c ; 
}

/**
NP::Combine
------------

Combines 2d arrays with different item counts using the largest item count plus one
for the middle dimension of the resulting 3d array.

For example a combination of 2d arrays with shapes: (n0,m) (n1,m) (n2,m) (n3,m) (n4,m) 
yields an output 3d array with shape: (5, 1+max(n0,n1,n2,n3,n4), m ) 
The extra "1+" column is used for including annotation of the n0, n1, n2, n3, n4  values
within the output array.  

The canonical usage is for combination of paired properties with m=2 however
the implementation could easily be generalized to work with higher dimensions if necessary.  

Note that if the n0,n1,n2,... dimensions are very different then the combined array will 
be inefficient with lots of padding so it makes sense to avoid large differences.  
When all the n are equal the annotation and adding could be disabled by setting annotate=false.  

See also:

tests/NPInterp.py:np_irregular_combine 
    python prototype 

test/NPCombineTest.cc 
    testing this NP::Combine and NP::interp on the combined array 

**/
inline NP* NP::Combine(const std::vector<const NP*>& aa, bool annotate)  // static 
{
    assert( aa.size() > 0 ); 
    const NP* a0 = aa[0] ; 

    const char* dtype0 = a0->dtype ; 
    int ebyte0 = a0->ebyte ; 
    unsigned ndim0 = a0->shape.size() ; 
    unsigned ldim0 = a0->shape[ndim0-1] ; 
    unsigned fdim_mx = a0->shape[0] ; 

    for(unsigned i=1 ; i < aa.size() ; i++)
    { 
        const NP* a = aa[i]; 
        assert( strcmp( a->dtype, dtype0 ) == 0 ); 

        unsigned ndim = a->shape.size() ; 
        assert( ndim == ndim0 && "input arrays must all have an equal number of dimensions" ); 

        unsigned ldim = a->shape[ndim-1] ; 
        assert( ldim == ldim0 && "last dimension of the input arrays must be equal" ); 

        unsigned fdim = a->shape[0] ; 
        if( fdim > fdim_mx ) fdim_mx = fdim ; 
    }
    unsigned width = fdim_mx + unsigned(annotate) ; 
    assert( ldim0 == 2 && "last dimension must currently be 2"); 

    NP* c = new NP(a0->dtype, aa.size(), width, ldim0 ); 
    unsigned item_bytes = c->item_bytes(); 

    std::cout 
        << "NP::Combine"
        << " ebyte0 " << ebyte0
        << " item_bytes " << item_bytes
        << " aa.size " << aa.size()
        << " width " << width
        << " ldim0 " << ldim0
        << " c " << c->desc() 
        << std::endl 
        ; 

    assert( item_bytes % ebyte0 == 0 ); 
    unsigned offset_bytes = 0 ; 
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        unsigned a_bytes = a->arr_bytes() ; 
        memcpy( c->data.data() + offset_bytes ,  a->data.data(),  a_bytes ); 
        offset_bytes += item_bytes ;  
    }
   
    if( annotate )
    {
        if( ebyte0 == 4 )
        {
            float* cc = c->values<float>();  
            UIF32 uif32 ;  
            for(unsigned i=0 ; i < aa.size() ; i++)
            {
                const NP* a = aa[i]; 
                uif32.u = a->shape[0] ;                  
                std::cout << " annotate " << i << " uif32.u  " << uif32.u  << std::endl ; 
                *(cc + (i+1)*item_bytes/ebyte0 - 1) = uif32.f ;   
            }  
        }
        else if( ebyte0 == 8 )
        {
            double* cc = c->values<double>();  
            UIF64 uif64 ;  
            for(unsigned i=0 ; i < aa.size() ; i++)
            {
                const NP* a = aa[i]; 
                uif64.u = a->shape[0] ;                  
                std::cout << " annotate " << i << " uif64.u  " << uif64.u  << std::endl ; 
                *(cc + (i+1)*item_bytes/ebyte0 - 1) = uif64.f ;   
            }  
        }
    }
    return c ; 
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

inline NP* NP::Load(const char* dir, const char* reldir, const char* name)
{
    std::string path = form_path(dir, reldir, name); 
    return Load(path.c_str());
}


inline std::string NP::form_name(const char* stem, const char* ext)
{
    std::stringstream ss ; 
    ss << stem ; 
    ss << ext ; 
    return ss.str(); 
}
inline std::string NP::form_path(const char* dir, const char* name)
{
    std::stringstream ss ; 
    ss << dir ; 
    if(name) ss << "/" << name ; 
    return ss.str(); 
}

inline std::string NP::form_path(const char* dir, const char* reldir, const char* name)
{
    std::stringstream ss ; 
    ss << dir ; 
    if(reldir) ss << "/" << reldir ; 
    if(name) ss << "/" << name ; 
    return ss.str(); 
}

inline void NP::clear()
{
    data.clear(); 
    data.shrink_to_fit(); 
    shape[0] = 0 ;  
}

inline bool NP::Exists(const char* dir, const char* name) // static 
{
    std::string path = form_path(dir, name); 
    return Exists(path.c_str()); 
}
inline bool NP::Exists(const char* path) // static 
{
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    return fp.fail() ? false : true ; 
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
        std::cerr << "NP::load Failed to load from path " << path << std::endl ; 
        return 1 ; 
    }

    std::getline(fp, _hdr );   
    _hdr += '\n' ; 

    decode_header(); 

    fp.read(bytes(), arr_bytes() );

    load_meta( path ); 

    return 0 ; 
}

inline int NP::load_meta( const char* path )
{
    std::string metapath = U::ChangeExt(path, ".npy", ".txt"); 
    std::ifstream fp(metapath.c_str(), std::ios::in);
    if(fp.fail()) return 1 ; 

    std::stringstream ss ;                       
    std::string line ; 
    while (std::getline(fp, line))
    {
        ss << line << std::endl ;   // getline swallows new lines  
    }
    meta = ss.str(); 
    return 0 ; 
}


inline void NP::save_header(const char* path)
{
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
}

inline void NP::old_save(const char* path)  // non-const due to update_headers
{
    std::cout << "NP::save path [" << path  << "]" << std::endl ; 
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
    stream.write( bytes(), arr_bytes() );
}

inline void NP::save(const char* path) const 
{
    std::cout << "NP::save path [" << path  << "]" << std::endl ; 
    std::string hdr = make_header(); 
    std::ofstream fpa(path, std::ios::out|std::ios::binary);
    fpa << hdr ; 
    fpa.write( bytes(), arr_bytes() );

    if( not meta.empty() )
    {
        std::string metapath = U::ChangeExt(path, ".npy", ".txt"); 
        std::cout << "NP::save metapath [" << metapath  << "]" << std::endl ; 
        std::ofstream fpm(metapath.c_str(), std::ios::out);
        fpm << meta ;  
    }  
}

inline void NP::save(const char* dir, const char* reldir, const char* name) const 
{
    std::cout << "NP::save dir [" << ( dir ? dir : "-" )  << "] reldir [" << ( reldir ? reldir : "-" )  << "] name [" << name << "]" << std::endl ; 
    std::string path = form_path(dir, reldir, name); 
    save(path.c_str()); 
}

inline void NP::save(const char* dir, const char* name) const 
{
    std::string path = form_path(dir, name); 
    save(path.c_str()); 
}

inline void NP::save_jsonhdr(const char* path) const 
{
    std::string json = make_jsonhdr(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << json ; 
}

inline void NP::save_jsonhdr(const char* dir, const char* name) const 
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

inline void NP::save_jsonhdr() const 
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

    const T* vv = cvalues<T>(); 

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
        << "meta:[" << meta << "]"
        << std::endl
        ; 
}


template <typename T> void NP::read(const T* data) 
{
    T* v = values<T>(); 

    NPS sh(shape); 
    for(int i=0 ; i < sh.ni_() ; i++ ) 
    for(int j=0 ; j < sh.nj_() ; j++ )
    for(int k=0 ; k < sh.nk_() ; k++ )
    for(int l=0 ; l < sh.nl_() ; l++ )
    for(int m=0 ; m < sh.nm_() ; m++ )
    {  
        int index = sh.idx(i,j,k,l,m); 
        *(v + index) = *(data + index ) ; 
    }   
}

template <typename T> void NP::read2(const T* data) 
{
    assert( sizeof(T) == ebyte ); 
    memcpy( bytes(), data, arr_bytes() );    
}


template <typename T> NP* NP::Linspace( T x0, T x1, unsigned nx )
{
    assert( x1 > x0 ); 
    assert( nx > 1 ) ; 
    NP* dom = NP::Make<T>(nx); 
    T* vv = dom->values<T>(); 
    for(unsigned i=0 ; i < nx ; i++) vv[i] = x0 + (x1-x0)*T(i)/T(nx-1) ;
    return dom ; 
}

/**
NP::MakeUniform
----------------

Create array of uniform random numbers between 0 and 1 using std::mt19937_64

**/

template <typename T> NP* NP::MakeUniform(unsigned ni, unsigned seed) // static 
{
    std::mt19937_64 rng;
    rng.seed(seed); 
    std::uniform_real_distribution<T> unif(0, 1);

    NP* uu = NP::Make<T>(ni); 
    T* vv = uu->values<T>(); 
    for(unsigned i=0 ; i < ni ; i++) vv[i] = unif(rng) ; 
    return uu ; 
}


template <typename T> unsigned NP::NumSteps( T x0, T x1, T dx )
{
    assert( x1 > x0 ); 
    assert( dx > T(0.) ) ; 

    unsigned ns = 0 ; 
    for(T x=x0 ; x <= x1 ; x+=dx ) ns+=1 ; 
    return ns ; 
}


template <typename T> NP* NP::Make( int ni_, int nj_, int nk_, int nl_, int nm_ )
{
    std::string dtype = descr_<T>::dtype() ; 
    NP* a = new NP(dtype.c_str(), ni_,nj_,nk_,nl_,nm_) ;    
    return a ; 
}



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
    a.read(data); 
    a.save(dir, name); 
}

// TODO: eliminate duplication between these methods

template <typename T> void NP::Write(const char* dir, const char* reldir, const char* name, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_ ) // static
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
        << " dir " << std::setw(50) << ( dir ? dir : "-" )
        << " reldir " << std::setw(50) << ( reldir ? reldir : "-" )
        << " name " << name
        << std::endl 
        ;   

    NP a(dtype.c_str(), ni_,nj_,nk_,nl_,nm_) ;    
    a.read(data); 
    a.save(dir, reldir, name); 
}



template <typename T> void NP::Write(const char* path, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_ ) // static
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
        << " path " << path
        << std::endl 
        ;   

    NP a(dtype.c_str(), ni_,nj_,nk_,nl_,nm_) ;    
    a.read(data); 
    a.save(path); 
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
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    WriteNames(path.c_str(), names, num_names_ ); 
}

inline void NP::WriteNames(const char* dir, const char* reldir, const char* name, const std::vector<std::string>& names, unsigned num_names_ )
{
    std::stringstream ss ; 
    ss << dir << "/" ;
    if(reldir) ss << reldir << "/" ; 
    ss << name ; 
    std::string path = ss.str() ; 
    WriteNames(path.c_str(), names, num_names_ ); 
}


inline void NP::WriteNames(const char* path, const std::vector<std::string>& names, unsigned num_names_ )
{
    unsigned num_names = num_names_ == 0 ? names.size() : num_names_ ; 
    assert( num_names <= names.size() ); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    for( unsigned i=0 ; i < num_names ; i++) stream << names[i] << std::endl ; 
    stream.close(); 
}

inline void NP::ReadNames(const char* dir, const char* name, std::vector<std::string>& names )
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    ReadNames(path.c_str(), names); 
}
inline void NP::ReadNames(const char* path, std::vector<std::string>& names )
{
    std::ifstream ifs(path);
    std::string line;
    while(std::getline(ifs, line)) names.push_back(line) ; 
}



#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>

#include "NPU.hh"
#include "net_hdr.hh"

/**
NP 
===

This replaces a templated struct approach which is in old/NP_old.hh.
Templating was however not convenient for streaming 
because you do not know the type until parsing the hdr. 
Plus most methods do not depend on the type... so this new NP
has a few templated methods rather than being a templated struct.

**/

struct NP
{
    NP(const char* dtype_="<f4", int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 
    void set_dtype(const char* dtype_); // may change shape and size of array while retaining the same underlying bytes 

    static void sizeof_check(); 
    static NP* Load(const char* path); 
    static NP* Load(const char* dir, const char* name); 
    static NP* MakeDemo(const char* dtype="<f4" , int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1 ); 

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

std::ostream& operator<<(std::ostream &os,  const NP& a) 
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

std::istream& operator>>(std::istream& is, NP& a)     
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

void NP::update_headers()
{
    std::string net_hdr = make_prefix(); 
    _prefix.assign(net_hdr.data(), net_hdr.length()); 

    std::string hdr =  make_header(); 
    _hdr.resize(hdr.length()); 
    _hdr.assign(hdr.data(), hdr.length()); 
}


std::string NP::make_header() const 
{
    std::string hdr =  NPU::_make_header( shape, dtype ) ;
    return hdr ; 
}
std::string NP::make_prefix() const 
{
    std::vector<unsigned> parts ;
    parts.push_back(hdr_bytes());
    parts.push_back(arr_bytes());
    parts.push_back(meta_bytes());
    parts.push_back(0);    // xxd neater to have 16 byte prefix 

    std::string net_hdr = net_hdr::pack( parts ); 
    return net_hdr ; 
}
std::string NP::make_jsonhdr() const 
{
    std::string json = NPU::_make_jsonhdr( shape, dtype ) ; 
    return json ; 
}  


NP* NP::MakeDemo(const char* dtype, int ni, int nj, int nk, int nl, int nm )
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
bool NP::decode_prefix()  
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
unsigned NP::prefix_size(unsigned index) const { return net_hdr::unpack(_prefix, index); }  

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

bool NP::decode_header()  
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

void NP::set_dtype(const char* dtype_)
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

unsigned NP::hdr_bytes() const { return _hdr.length() ; }
unsigned NP::num_values() const { return NPS::size(shape) ;  }
unsigned NP::arr_bytes()  const { return NPS::size(shape)*ebyte ; }
unsigned NP::meta_bytes() const { return meta.length() ; }

char*        NP::bytes() { return (char*)data.data() ;  } 
const char*  NP::bytes() const { return (char*)data.data() ;  } 


NP::NP(const char* dtype_, int ni, int nj, int nk, int nl, int nm )
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




template<typename T> T*  NP::values() { return (T*)data.data() ;  } 

template<typename T> void NP::fill(T value)
{
    T* vv = values<T>(); 
    for(unsigned i=0 ; i < size ; i++) *(vv+i) = value ; 
}

template<typename T> void NP::_fillIndexFlat(T offset)
{
    T* vv = values<T>(); 
    for(unsigned i=0 ; i < size ; i++) *(vv+i) = T(i) + offset ; 
}


void NP::sizeof_check() // static 
{
    assert( sizeof(float) == 4  );  
    assert( sizeof(double) == 8  );  

    assert( sizeof(char) == 1 );  
    assert( sizeof(short) == 2 );
    assert( sizeof(int)   == 4 );
    assert( sizeof(long)  == 8 );
    assert( sizeof(long long)  == 8 );
}

void NP::fillIndexFlat()
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


void NP::dump(int i0, int i1) const 
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

std::string NP::desc() const 
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

NP* NP::Load(const char* path)
{
    NP* a = new NP() ; 
    int rc = a->load(path) ; 
    return rc == 0 ? a  : NULL ; 
}

NP* NP::Load(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    return Load(path.c_str());
}

std::string NP::form_path(const char* dir, const char* name)
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    return ss.str(); 
}

int NP::load(const char* dir, const char* name)
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

int NP::load(const char* path)
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

void NP::save_header(const char* path)
{
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
}

void NP::save(const char* path)
{
    update_headers(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << _hdr ; 
    stream.write( bytes(), arr_bytes() );
}

void NP::save(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    save(path.c_str()); 
}

void NP::save_jsonhdr(const char* path)
{
    std::string json = make_jsonhdr(); 
    std::ofstream stream(path, std::ios::out|std::ios::binary);
    stream << json ; 
}

void NP::save_jsonhdr(const char* dir, const char* name)
{
    std::string path = form_path(dir, name); 
    save_jsonhdr(path.c_str()); 
}

std::string NP::get_jsonhdr_path() const 
{
    assert( lpath.empty() == false ); 
    assert( U::EndsWith(lpath.c_str(), ".npy" ) ); 
    std::string path = U::ChangeExt(lpath.c_str(), ".npy", ".npj"); 
    return path ; 
}

void NP::save_jsonhdr()
{
    std::string path = get_jsonhdr_path() ; 
    std::cout << "NP::save_jsonhdr to " << path << std::endl  ; 
    save_jsonhdr(path.c_str()); 
}


template <typename T> std::string NP::_present(T v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << v  ;      
    return ss.str();
}

// needs specialization to _present char as an int rather than a character
template<>  std::string NP::_present(char v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << int(v)  ;      
    return ss.str();
}
template<>  std::string NP::_present(unsigned char v) const
{
    std::stringstream ss ; 
    ss << " " << std::fixed << std::setw(8) << unsigned(v)  ;      
    return ss.str();
}
template<>  std::string NP::_present(float v) const
{
    std::stringstream ss ; 
    ss << " " << std::setw(10) << std::fixed << std::setprecision(3) << v ;
    return ss.str();
}
template<>  std::string NP::_present(double v) const
{
    std::stringstream ss ; 
    ss << " " << std::setw(10) << std::fixed << std::setprecision(3) << v ;
    return ss.str();
}


template <typename T> void NP::_dump(int i0_, int i1_) const 
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
template<> const T* NP::values<T>() const { return (T*)data.data() ; }
template<>       T* NP::values<T>()      {  return (T*)data.data() ; }
template       void NP::_fillIndexFlat<T>(T) ;

EOC
}
specialize(){ specialize-head ; specialize-types | while read t ; do specialize- "$t" ; done  ; }
specialize

**/


// template specializations generated by below bash functions

template<> const float* NP::values<float>() const { return (float*)data.data() ; }
template<>       float* NP::values<float>()      {  return (float*)data.data() ; }
template       void NP::_fillIndexFlat<float>(float) ;

template<> const double* NP::values<double>() const { return (double*)data.data() ; }
template<>       double* NP::values<double>()      {  return (double*)data.data() ; }
template       void NP::_fillIndexFlat<double>(double) ;

template<> const char* NP::values<char>() const { return (char*)data.data() ; }
template<>       char* NP::values<char>()      {  return (char*)data.data() ; }
template       void NP::_fillIndexFlat<char>(char) ;

template<> const short* NP::values<short>() const { return (short*)data.data() ; }
template<>       short* NP::values<short>()      {  return (short*)data.data() ; }
template       void NP::_fillIndexFlat<short>(short) ;

template<> const int* NP::values<int>() const { return (int*)data.data() ; }
template<>       int* NP::values<int>()      {  return (int*)data.data() ; }
template       void NP::_fillIndexFlat<int>(int) ;

template<> const long* NP::values<long>() const { return (long*)data.data() ; }
template<>       long* NP::values<long>()      {  return (long*)data.data() ; }
template       void NP::_fillIndexFlat<long>(long) ;

template<> const long long* NP::values<long long>() const { return (long long*)data.data() ; }
template<>       long long* NP::values<long long>()      {  return (long long*)data.data() ; }
template       void NP::_fillIndexFlat<long long>(long long) ;

template<> const unsigned char* NP::values<unsigned char>() const { return (unsigned char*)data.data() ; }
template<>       unsigned char* NP::values<unsigned char>()      {  return (unsigned char*)data.data() ; }
template       void NP::_fillIndexFlat<unsigned char>(unsigned char) ;

template<> const unsigned short* NP::values<unsigned short>() const { return (unsigned short*)data.data() ; }
template<>       unsigned short* NP::values<unsigned short>()      {  return (unsigned short*)data.data() ; }
template       void NP::_fillIndexFlat<unsigned short>(unsigned short) ;

template<> const unsigned int* NP::values<unsigned int>() const { return (unsigned int*)data.data() ; }
template<>       unsigned int* NP::values<unsigned int>()      {  return (unsigned int*)data.data() ; }
template       void NP::_fillIndexFlat<unsigned int>(unsigned int) ;

template<> const unsigned long* NP::values<unsigned long>() const { return (unsigned long*)data.data() ; }
template<>       unsigned long* NP::values<unsigned long>()      {  return (unsigned long*)data.data() ; }
template       void NP::_fillIndexFlat<unsigned long>(unsigned long) ;

template<> const unsigned long long* NP::values<unsigned long long>() const { return (unsigned long long*)data.data() ; }
template<>       unsigned long long* NP::values<unsigned long long>()      {  return (unsigned long long*)data.data() ; }
template       void NP::_fillIndexFlat<unsigned long long>(unsigned long long) ;


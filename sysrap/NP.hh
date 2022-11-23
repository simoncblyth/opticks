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
    static constexpr const char* EXT = ".npy" ; 
#ifdef WITH_VERBOSE
    static const bool VERBOSE = true ; 
#else
    static const bool VERBOSE = false ; 
#endif

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

    static bool StartsWith( const char* s, const char* q) ; 
    static bool Contains(   const char* s, const char* q) ; 
    static NP*  MakeValues( const std::vector<std::pair<std::string, double>>& values, const char* contains=nullptr ); 
    std::string descValues() const ; 

    template<typename T> static NP*  Make( int ni_=-1, int nj_=-1, int nk_=-1, int nl_=-1, int nm_=-1, int no_=-1 );  // dtype from template type
    template<typename T> static NP*  Linspace( T x0, T x1, unsigned nx, int npayload=-1 ); 
    template<typename T> static NP*  MakeDiv( const NP* src, unsigned mul  ); 
    template<typename T> static NP*  Make( const std::vector<T>& src ); 
    
    template<typename T, typename... Args> static NP*  Make(const T* src, Args ... shape );  // Make_ellipsis


    template<typename T> static NP*  Make( T d0, T v0, T d1, T v1 ); 
    template<typename T> static T To( const char* a ); 
    template<typename T> static bool ConvertsTo( const char* a ); 
    template<typename T> static NP* FromString(const char* str, char delim=' ') ;  


    template<typename T> static unsigned NumSteps( T x0, T x1, T dx ); 

    // ctor
    NP(const char* dtype_, const std::vector<int>& shape_ ); 
    NP(const char* dtype_="<f4", int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 

    void init(); 
    void set_shape( int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1);  
    void set_shape( const std::vector<int>& src_shape ); 
    // CAUTION: DO NOT USE *set_shape* TO CHANGE SHAPE (as it calls *init*) INSTEAD USE *change_shape* 
    bool has_shape(int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ) const ;  
    void change_shape(int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ) ;   // one dimension entry left at -1 can be auto-set

    void change_shape_to_3D() ; 

    void set_dtype(const char* dtype_); // *set_dtype* may change shape and size of array while retaining the same underlying bytes 

    static void sizeof_check(); 

    template<typename T> static int DumpCompare( const NP* a, const NP* b, unsigned a_column, unsigned b_column, const T epsilon ); 

    static int Memcmp( const NP* a, const NP* b ); 
    static NP* Concatenate(const std::vector<NP*>& aa); 
    static NP* Concatenate(const char* dir, const std::vector<std::string>& names); 

    static NP* Combine(const std::vector<const NP*>& aa, bool annotate=true); 
    template<typename... Args> static NP* Combine(Args ... aa);  // Combine_ellipsis

    



    // load array asis 
    static NP* Load(const char* path); 
    static NP* Load_(const char* path); 
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


    template<typename T> static NP* MakeFlat(int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 
    static NP* MakeDemo(const char* dtype="<f4" , int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 

    template<typename T> static void Write(const char* dir, const char* name, const std::vector<T>& values ); 
    //template<typename T> static void Read( const char* dir, const char* name,       T* dst ); 


    template<typename T> static void Write(const char* dir, const char* name, const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 
    template<typename T> static void Write(const char* dir, const char* reldir, const char* name, const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 

    template<typename T> static void Write(const char* path                 , const T* data, int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 

    static void WriteNames(const char* dir, const char* name,                     const std::vector<std::string>& names, unsigned num_names=0, bool append=false ); 
    static void WriteNames(const char* dir, const char* reldir, const char* name, const std::vector<std::string>& names, unsigned num_names=0, bool append=false ); 
    static void WriteNames(const char* path,                                      const std::vector<std::string>& names, unsigned num_names=0, bool append=false ); 

    static void ReadNames(const char* dir, const char* name, std::vector<std::string>& names ) ;
    static void ReadNames(const char* path,                  std::vector<std::string>& names ) ;


    template<typename T>
    static void ReadKV(const char* dir, const char* name, 
                      std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extra=nullptr ) ;

    template<typename T>
    static void ReadKV(const char* path, 
                       std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extra=nullptr ) ;


    template<typename T>
    static T ReadKV_Value(const char* dir, const char* name, const char* key ); 

    template<typename T>
    static T ReadKV_Value(const char* spec_or_path, const char* key ); 


    template<typename T>
    static std::string DescKV(const std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extra); 

    template <typename T> 
    static NP* LoadFromTxtFile(const char* path); 

    template <typename T> 
    static NP* LoadFromTxtFile(const char* base, const char* relp); 


    template <typename T> 
    static NP* ZEROProp(); 

    template <typename T> 
    static NP* LoadFromString(const char* str, const char* path_for_debug_messages=nullptr ); 

    // FindUnit returns last matching unit string, so more specific strings that contain earlier 
    // ones should come later in list 
    static constexpr const char* UNITS = "eV MeV nm mm cm m ns g/cm2/MeV" ; 
    static char* FindUnit(const char* line, const std::vector<std::string>& units  );  
    static void GetUnits(std::vector<std::string>& units ); 
    static bool IsListed(const std::vector<std::string>& ls, const char* str); 
    static std::string StringConcat(const std::vector<std::string>& ls, char delim=' ' ); 

    static unsigned CountChar(const char* str, char q ); 
    static void ReplaceCharInsitu(       char* str, char q, char n, bool first ); 
    static const char* ReplaceChar(const char* str, char q, char n, bool first ); 

    static const char* Resolve( const char* spec) ; 
    static const char* ResolveProp(const char* spec); 


    static void        WriteString( const char* dir, const char* reldir, const char* name, const char* str ); 
    static void        WriteString( const char* dir, const char* name, const char* str ); 
    static void        WriteString( const char* path, const char* str ); 

    static const char* ReadString( const char* dir, const char* reldir, const char* name);
    static const char* ReadString( const char* dir, const char* name);
    static const char* ReadString( const char* path );

    static const char* ReadString2( const char* path );


    template<typename T> T*       values() ; 
    template<typename T> const T*  cvalues() const  ; 

    unsigned  index(  int i,  int j=0,  int k=0,  int l=0, int m=0, int o=0) const ; 
    unsigned  index0( int i,  int j=-1,  int k=-1,  int l=-1, int m=-1, int o=-1) const ; 

    unsigned dimprod(unsigned q) const ;    // product of dimensions starting from dimension q

    template<typename... Args> 
    unsigned index_(Args ... idxx ) const ; 

    template<typename... Args> 
    unsigned stride_(Args ... idxx ) const ; 

    template<typename... Args> 
    unsigned offset_(Args ... idxx ) const ; 


    template<typename T>
    static std::string ArrayString(const std::vector<T>& vec, unsigned modulo=10 ); 

    template<typename T, typename... Args> 
    std::string sliceArrayString(Args ... idxx ) const ;  

    // use -1 to mark the last dimension to select upon 
    // eg to select first item use (0, -1) 
    template<typename T, typename... Args> 
    void slice(std::vector<T>& out, Args ... idxx ) const ;  // slice_ellipsis


    template<typename T> 
    void slice_(std::vector<T>& out, const std::vector<int>& idxx ) const ; 

    template<typename T> 
    static std::string DescSlice(const std::vector<T>& out, unsigned edge ); 

    template<typename T> 
    static std::string DescSliceBrief(const std::vector<T>& out); 


    static std::string DescIdx(const std::vector<int>& idxx ); 


    int pickdim__(    const std::vector<int>& idxx) const ; 

    unsigned index__( const std::vector<int>& idxx) const ; 
    unsigned stride__(const std::vector<int>& idxx) const ; 
    unsigned offset__(const std::vector<int>& idxx) const ; 



    unsigned  itemsize_(int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1) const ; 
    void      itembytes_(const char** start,  unsigned& num_bytes, int i=-1, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1 ) const  ; 

    template<typename T> T           get( int i,  int j=0,  int k=0,  int l=0, int m=0, int o=0) const ; 
    template<typename T> void        set( T val, int i,  int j=0,  int k=0,  int l=0, int m=0, int o=0 ) ; 

    template<typename T> bool is_allzero() const ; 
    template<typename T> void fill(T value); 
    template<typename T> void _fillIndexFlat(T offset=0); 
    template<typename T> void _dump(int i0=-1, int i1=-1, int j0=-1, int j1=-1) const ;   


    static NP* MakeLike(  const NP* src);  
    static void CopyMeta( NP* b, const NP* a ); 

    static constexpr const char* Preserve_Last_Column_Integer_Annotation = "Preserve_Last_Column_Integer_Annotation" ; 
    void set_preserve_last_column_integer_annotation(); 
    bool is_preserve_last_column_integer_annotation() const ; 
    static float PreserveNarrowedDoubleInteger( double f ); 

    static NP* MakeNarrow(const NP* src); 
    static NP* MakeWide(  const NP* src); 
    static NP* MakeCopy(  const NP* src); 
    static NP* MakeCopy3D(const NP* src); 
    static NP* ChangeShape3D(NP* src); 

    static NP* MakeWideIfNarrow(  const NP* src); 
    static NP* MakeNarrowIfWide(  const NP* src); 

    template<typename T>
    static NP* MakeWithType(const NP* src); 


    template<typename... Args> 
    static NP* MakeSelectCopy(  const NP* src, Args ... items );  // MakeSelectCopy_ellipsis

    static NP* MakeSelectCopyE_( const NP* src, const char* ekey, const char* fallback=nullptr, char delim=',' ); 
    static NP* MakeSelectCopy_( const NP* src, const char* items ); 
    static NP* MakeSelectCopy_( const NP* src, const std::vector<int>* items ); 
    static NP* MakeSelectCopy_( const NP* src, const int* items, int num_items ); 


    static NP* MakeItemCopy(  const NP* src, int i,int j=-1,int k=-1,int l=-1,int m=-1, int o=-1 ); 
    void  item_shape(std::vector<int>& sub, int i, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1 ) const ; 
    NP*   spawn_item(  int i, int j=-1, int k=-1, int l=-1, int m=-1, int o=-1  ) const ; 

    template<typename T> static NP* MakeCDF(  const NP* src );
    template<typename T> static NP* MakeICDF(  const NP* src, unsigned nu, unsigned hd_factor, bool dump );
    template<typename T> static NP* MakeProperty(const NP* a, unsigned hd_factor ); 
    template<typename T> static NP* MakeLookupSample(const NP* icdf_prop, unsigned ni, unsigned seed=0u, unsigned hd_factor=0u );  
    template<typename T> static NP* MakeUniform( unsigned ni, unsigned seed=0u );  


 
    NP* copy() const ; 

    bool is_pshaped() const ; 
    template<typename T> T    plhs(unsigned column ) const ;  
    template<typename T> T    prhs(unsigned column ) const ;  
    template<typename T> int  pfindbin(const T value, unsigned column, bool& in_range ) const ;  
    template<typename T> void get_edges(T& lo, T& hi, unsigned column, int ibin) const ; 


    template<typename T> T    psum(unsigned column ) const ;  
    template<typename T> void pscale(T scale, unsigned column);
    template<typename T> void pscale_add(T scale, T add, unsigned column);
    template<typename T> void pdump(const char* msg="NP::pdump", T d_scale=1., T v_scale=1.) const ; 
    template<typename T> void minmax(T& mn, T&mx, unsigned j=1, int item=-1 ) const ; 
    template<typename T> void linear_crossings( T value, std::vector<T>& crossings ) const ; 
    template<typename T> NP*  trapz() const ;                      // composite trapezoidal integration, requires pshaped

    template<typename T> void psplit(std::vector<T>& domain, std::vector<T>& values) const ; 
    template<typename T> T    pdomain(const T value, int item=-1, bool dump=false  ) const ; 
    template<typename T> T    interp(T x, int item=-1) const ;                  // requires pshaped 
    template<typename T> T    interp2D(T x, T y, int item=-1) const ;   


    template<typename T> T    interpHD(T u, unsigned hd_factor, int item=-1 ) const ; 

    template<typename T> T    interp(int iprop, T x) const ;           // deprecated signature for combined_interp  
    template<typename T> T    combined_interp_3(int i,               T x) const ;  // requires NP::Combine of pshaped arrays 
    template<typename T> T    combined_interp_5(int i, int j, int k, T x) const ;  // requires NP::Combine of pshapes arrays 


    template<typename T> T    _combined_interp(const T* vv, unsigned niv, T x) const  ; 


    template<typename T> NP*  cumsum(int axis=0) const ; 
    template<typename T> void divide_by_last() ; 



    template<typename T> void read(const T* src);
    template<typename T> void read2(const T* src);
    template<typename T> void write(T* dst) const ; 

    template<typename T> std::string _present(T v) const ; 

    void fillIndexFlat(); 
    void dump(int i0=-1, int i1=-1, int j0=-1, int j1=-1) const ; 


    static bool Exists(const char* base, const char* rel, const char* name);   
    static bool Exists(const char* dir, const char* name);   
    static bool Exists(const char* path);   
    int load(const char* dir, const char* name);   
    int load(const char* path);   

    int load_string_(  const char* path, const char* ext, std::string& str ); 
    int load_strings_( const char* path, const char* ext, std::vector<std::string>& vstr ); 
    int load_meta(  const char* path ); 
    int load_names( const char* path ); 

    void save_string_( const char* path, const char* ext, const std::string& str ) const ; 
    void save_strings_(const char* path, const char* ext, const std::vector<std::string>& vstr ) const ; 
    void save_meta( const char* path) const ;  
    void save_names(const char* path) const ;  

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
    std::string brief() const ; 
    std::string sstr() const ; 


    void set_meta( const std::vector<std::string>& lines, char delim='\n' ); 
    void get_meta( std::vector<std::string>& lines,       char delim='\n' ) const ; 

    void set_names( const std::vector<std::string>& lines ) ; 
    void get_names( std::vector<std::string>& lines ) const ; 

    int  get_name_index( const char* qname ) const ;  
    int  get_name_index( const char* qname, unsigned& count ) const ;  
    static int NameIndex( const char* qname, unsigned& count, const std::vector<std::string>& names ); 
    
    bool is_named_shape() const ; 
    template<typename T> T  get_named_value( const char* qname, T fallback ) const ; 


    static std::string               get_meta_string_(const char* metadata, const char* key);  
    static std::string               get_meta_string( const std::string& meta, const char* key) ;  

    bool has_meta() const ; 
    template<typename T> static T    get_meta_(const char* metadata, const char* key, T fallback=0) ;  // for T=std::string must set fallback to ""
    template<typename T> T    get_meta(const char* key, T fallback=0) const ;  // for T=std::string must set fallback to ""
    template<typename T> void set_meta(const char* key, T value ) ;  

    template<typename T> static T    GetMeta( const std::string& mt, const char* key, T fallback ); 
    template<typename T> static void SetMeta(       std::string& mt, const char* key, T value ); 

    std::string descMeta() const ; 
    const char* get_lpath() const ; 

    char*       bytes();  
    const char* bytes() const ;  

    unsigned num_items() const ;       // shape[0] 
    unsigned num_values() const ;      // all values, product of shape[0]*shape[1]*...
    unsigned num_itemvalues() const ;  // values after first dimension 
    unsigned arr_bytes() const ;       // formerly num_bytes
    unsigned item_bytes() const ;      // *item* comprises all dimensions beyond the first 
    unsigned hdr_bytes() const ;  
    unsigned meta_bytes() const ;


    void clear() ; 
  
    // primary data members 
    std::vector<char> data ; 
    std::vector<int>  shape ; 
    std::string       meta ; 
    std::vector<std::string>  names ;  // CHANGED to vector of string for convenience of reference passing, eg for CSGName

    // non-persisted transients, set on loading 
    std::string lpath ; 
    std::string lfold ; 

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


inline void NP::clear()
{
    data.clear(); 
    data.shrink_to_fit(); 
    shape[0] = 0 ; 
}


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

    if(NP::VERBOSE) std::cout 
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

inline bool NP::StartsWith( const char* s, const char* q) // static
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ;
}
inline bool NP::Contains( const char* s, const char* q) // static
{
    return s && q && strlen(q) <= strlen(s) && strstr(s, q) != nullptr ;
}

inline NP* NP::MakeValues( const std::vector<std::pair<std::string, double>>& values, const char* contains ) // static
{
    if(VERBOSE) std::cout 
        << "NP::MakeValues values.size " << values.size() 
        << " contains " << ( contains ? contains : "-" )
        << std::endl 
        ;  

    std::vector<std::string> nams ; 
    std::vector<double> vals ; 

    for(unsigned i=0 ; i < values.size() ; i++)
    {   
        const std::pair<std::string, double>& kv = values[i] ; 
        const char* k = kv.first.c_str() ; 
        double v = kv.second ;

        bool select = contains == nullptr || Contains( k, contains ) ; 

        if(VERBOSE) std::cout 
            << "NP::MakeValues " 
            << std::setw(3) << i 
            << " v " << std::setw(10) << std::fixed << std::setprecision(4) << v 
            << " k " << std::setw(60) << k 
            << " select " << select 
            <<  std::endl 
            ;
 
        if(select)
        {   
            nams.push_back(k); 
            vals.push_back(v); 
        }   
    }  
    if(VERBOSE) std::cout << "NP::MakeValues vals.size " << vals.size() << std::endl ;  
    if(vals.size() == 0 ) return nullptr ; 

    NP* vv = NP::Make<double>( vals ) ; 
    vv->set_names( nams );  

    return vv ; 
}



template <typename T> inline T NP::To( const char* a )   // static 
{   
    std::string s(a);
    std::istringstream iss(s);
    T v ;   
    iss >> v ; 
    return v ; 
}

template <typename T> inline bool NP::ConvertsTo( const char* a )   // static 
{   
    if( a == nullptr ) return false ; 
    if( strlen(a) == 0) return false ; 
    std::string s(a);
    std::istringstream iss(s);
    T v ;   
    iss >> v ; 
    return iss.fail() == false ; 
}







// specialization for std::string as the above truncates at the first blank in the string, see tests/NP_set_meta_get_meta_test.cc  
template<> inline std::string NP::To(const char* a ) 
{
    std::string s(a); 
    return s ; 
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


inline NP* NP::MakeDemo(const char* dtype, int ni, int nj, int nk, int nl, int nm, int no )
{
    NP* a = new NP(dtype, ni, nj, nk, nl, nm, no);
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

    if(VERBOSE) std::cout 
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

    if(VERBOSE) std::cout 
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
inline unsigned NP::num_items() const { return shape[0] ;  }
inline unsigned NP::num_values() const { return NPS::size(shape) ;  }
inline unsigned NP::num_itemvalues() const { return NPS::itemsize(shape) ;  }
inline unsigned NP::arr_bytes()  const { return NPS::size(shape)*ebyte ; }
inline unsigned NP::item_bytes() const { return NPS::itemsize(shape)*ebyte ; }
inline unsigned NP::meta_bytes() const { return meta.length() ; }

inline char*        NP::bytes() { return (char*)data.data() ;  } 
inline const char*  NP::bytes() const { return (char*)data.data() ;  } 


inline NP::NP(const char* dtype_, const std::vector<int>& shape_ )
    :
    shape(shape_),
    dtype(strdup(dtype_)),
    uifc(NPU::_dtype_uifc(dtype)),
    ebyte(NPU::_dtype_ebyte(dtype)),
    size(NPS::size(shape))
{
    init(); 
}

inline NP::NP(const char* dtype_, int ni, int nj, int nk, int nl, int nm, int no )
    :
    dtype(strdup(dtype_)),
    uifc(NPU::_dtype_uifc(dtype)),
    ebyte(NPU::_dtype_ebyte(dtype)),
    size(NPS::set_shape(shape, ni,nj,nk,nl,nm,no ))
{
    init(); 
}

inline void NP::init()
{
#ifdef OLD
    int num_char = size*ebyte ; 
#else
    unsigned long long size_ = size ; 
    unsigned long long ebyte_ = ebyte ; 
    unsigned long long num_char = size_*ebyte_ ; 
#endif

#ifdef DEBUG
    std::cout 
        << "NP::init"
        << " size " << size
        << " ebyte " << ebyte
        << " num_char " << num_char 
        << std::endl 
        ;
#endif

    data.resize( num_char ) ;  // vector of char  
    std::fill( data.begin(), data.end(), 0 );     
    _prefix.assign(net_hdr::LENGTH, '\0' ); 
    _hdr = make_header(); 
}




inline void NP::set_shape(int ni, int nj, int nk, int nl, int nm, int no)
{
    size = NPS::copy_shape(shape, ni, nj, nk, nl, nm, no); 
    init(); 
}
inline void NP::set_shape(const std::vector<int>& src_shape)
{
    size = NPS::copy_shape(shape, src_shape); 
    init(); 
}

inline bool NP::has_shape(int ni, int nj, int nk, int nl, int nm, int no) const 
{
    unsigned ndim = shape.size() ; 
    return 
           ( ni == -1 || ( ndim > 0 && int(shape[0]) == ni)) && 
           ( nj == -1 || ( ndim > 1 && int(shape[1]) == nj)) && 
           ( nk == -1 || ( ndim > 2 && int(shape[2]) == nk)) && 
           ( nl == -1 || ( ndim > 3 && int(shape[3]) == nl)) && 
           ( nm == -1 || ( ndim > 4 && int(shape[4]) == nm)) && 
           ( no == -1 || ( ndim > 5 && int(shape[5]) == no))  
           ;
}


/**
NP::change_shape
------------------

One dimension can be -1 causing it to be filled automatically.
See tests/NPchange_shapeTest.cc

**/

inline void NP::change_shape(int ni, int nj, int nk, int nl, int nm, int no)
{
    int size2 = NPS::change_shape(shape, ni, nj, nk, nl, nm, no); 
    assert( size == size2 ); 
}

inline void NP::change_shape_to_3D() 
{
    unsigned ndim = shape.size() ; 
    if(VERBOSE) std::cerr << "NP::change_shape_to_3D sstr " << sstr() << std::endl ; 

    if( ndim < 3 ) 
    {   
        std::cerr << "NP::change_shape_to_3D : ndim < 3 : must be 3 or more, not: " << ndim << std::endl ; 
        assert(0); 
    }   
    else if( ndim == 3 ) 
    {   
        if(VERBOSE) std::cerr << "NP::change_shape_to_3D : ndim == 3, no reshaping needed " << std::endl ; 
    }   
    else if( ndim > 3 ) 
    {   
        if(VERBOSE) std::cerr << "NP::change_shape_to_3D : ndim > 3, reshaping needed, ndim: " << ndim  << std::endl ;   
        int ni = 1 ; 
        for(int i=0 ; i < int(ndim) - 2 ; i++) ni *= shape[i] ; 
        // scrunch up the higher dimensions          
        change_shape(ni, shape[ndim-2], shape[ndim-1] );  
        if(VERBOSE) std::cerr << "NP::change_shape_to_3D : changed shape to : " << sstr() << std::endl  ; 
    }   
}








template<typename T> inline const T*  NP::cvalues() const { return (T*)data.data() ;  } 
template<typename T> inline T*        NP::values() { return (T*)data.data() ;  } 

/**
NP::index
-----------

Provides the flat value index from a set of integer dimension indices. 
Negative dimension indices are interpreted to count from the back, ie -1 is the last element 
in a dimension. 

**/

inline unsigned NP::index( int i,  int j,  int k,  int l, int m, int o ) const 
{
    unsigned nd = shape.size() ; 
    unsigned ni = nd > 0 ? shape[0] : 1 ; 
    unsigned nj = nd > 1 ? shape[1] : 1 ; 
    unsigned nk = nd > 2 ? shape[2] : 1 ; 
    unsigned nl = nd > 3 ? shape[3] : 1 ; 
    unsigned nm = nd > 4 ? shape[4] : 1 ; 
    unsigned no = nd > 5 ? shape[5] : 1 ; 

    unsigned ii = i < 0 ? ni + i : i ; 
    unsigned jj = j < 0 ? nj + j : j ; 
    unsigned kk = k < 0 ? nk + k : k ; 
    unsigned ll = l < 0 ? nl + l : l ; 
    unsigned mm = m < 0 ? nm + m : m ; 
    unsigned oo = o < 0 ? no + o : o ; 

    return  ii*nj*nk*nl*nm*no + jj*nk*nl*nm*no + kk*nl*nm*no + ll*nm*no + mm*no + oo ;
}

/**
NP::index0 : Provides element offset 
---------------------------------------

Same as NP::index but -ve "missing" indices are treated as if they were zero. 

**/

inline unsigned NP::index0( int i,  int j,  int k,  int l, int m, int o) const 
{
    unsigned nd = shape.size() ; 

    unsigned ni = nd > 0 ? shape[0] : 1 ; 
    unsigned nj = nd > 1 ? shape[1] : 1 ; 
    unsigned nk = nd > 2 ? shape[2] : 1 ; 
    unsigned nl = nd > 3 ? shape[3] : 1 ; 
    unsigned nm = nd > 4 ? shape[4] : 1 ; 
    unsigned no = nd > 5 ? shape[5] : 1 ; 

    unsigned ii = i < 0 ? 0 : i ; 
    unsigned jj = j < 0 ? 0 : j ; 
    unsigned kk = k < 0 ? 0 : k ; 
    unsigned ll = l < 0 ? 0 : l ; 
    unsigned mm = m < 0 ? 0 : m ; 
    unsigned oo = o < 0 ? 0 : o ; 

    assert( ii < ni ); 
    assert( jj < nj ); 
    assert( kk < nk ); 
    assert( ll < nl ); 
    assert( mm < nm ); 
    assert( oo < no ); 

    /* 
    std::cout << " ii " << ii << " nj*nk*nl*nm*no " << std::setw(10) << nj*nk*nl*nm*no << " ii*nj*nk*nl*nm*no " << ii*nj*nk*nl*nm*no << std::endl ;
    std::cout << " jj " << jj << "    nk*nl*nm*no " << std::setw(10) <<    nk*nl*nm*no << " jj*   nk*nl*nm*no " << jj*nk*nl*nm*no    << std::endl ;
    std::cout << " kk " << kk << "       nl*nm*no " << std::setw(10) <<       nl*nm*no << " kk*      nl*nm*no " << kk*nl*nm*no       << std::endl ;
    std::cout << " ll " << ll << "          nm*no " << std::setw(10) <<          nm*no << " ll*         nm*no " << ll*nm*no          << std::endl ;
    std::cout << " mm " << mm << "             no " << std::setw(10) <<             no << " mm*            no " << mm*no             << std::endl ;
    std::cout << " oo " << oo << "              1 " << std::setw(10) <<              1 << " oo*             1 " << oo                << std::endl ;   
    */

    return  ii*nj*nk*nl*nm*no + jj*nk*nl*nm*no + kk*nl*nm*no + ll*nm*no + mm*no + oo ;
    //      i                   j                k             l          m       o 
}

inline unsigned NP::dimprod(unsigned q) const   // product of dimensions starting from dimension q
{
    unsigned dim = 1 ; 
    for(unsigned d=q ; d < shape.size() ; d++) dim *= shape[d] ; 
    return dim ;   
} 


template<typename... Args>
inline unsigned NP::index_(Args ... idxx_) const 
{
    std::vector<int> idxx = {idxx_...};
    return index__(idxx); 
}

template<typename... Args>
inline unsigned NP::stride_(Args ... idxx_) const 
{
    std::vector<int> idxx = {idxx_...};
    return stride__(idxx); 
}

template<typename... Args>
inline unsigned NP::offset_(Args ... idxx_) const 
{
    std::vector<int> idxx = {idxx_...};
    return offset__(idxx); 
}


template<typename T>
inline std::string NP::ArrayString(const std::vector<T>& vec, unsigned modulo ) // static
{
    const char* np_type = "uint64" ;   // TODO: make this depend on type
    unsigned size = vec.size(); 

    std::stringstream ss ; 
    ss << "np.array([ "  ;   
    for(unsigned i=0 ; i < size ; i++) 
    {   
        if( size > modulo && (( i % modulo ) == 0) ) ss << std::endl ; 
        ss << vec[i] << ( i < size - 1 ? ", " : " " ) ; 
    }   
    ss << "], dtype=np." << np_type << " )"  ; 

    std::string s = ss.str(); 
    return s ; 
}


template<typename T, typename... Args> 
inline std::string NP::sliceArrayString(Args ... idxx_ ) const 
{
    std::vector<int> idxx = {idxx_...};
    std::vector<T> out ; 
    slice(out, idxx ); 
    return ArrayString(out, 10); 
}


/**
NP::slice "slice_ellipsis"
---------------------------

**/

template<typename T, typename... Args> inline void NP::slice(std::vector<T>& out, Args ... idxx_ ) const 
{
   std::vector<int> idxx = {idxx_...};
   slice_(out, idxx); 
}








template<typename T> inline void NP::slice_(std::vector<T>& out, const std::vector<int>& idxx ) const 
{
    if(NP::VERBOSE) 
    std::cout 
        << " DescIdx(idxx) " << DescIdx(idxx)
        << " sstr() " << sstr() 
        << std::endl
        ;

    bool all_dim =  idxx.size() == shape.size() ; 
    if(!all_dim) std::cerr << " idxx.size " << idxx.size() << " shape.size " << shape.size() << " all_dim " << all_dim << std::endl ; 
    assert(all_dim) ; 

    int slicedim = pickdim__(idxx); 
    assert( slicedim > -1 ); 

    unsigned start = index__(idxx) ; 
    unsigned stride = stride__(idxx) ; 
    unsigned offset = offset__(idxx) ; 
    unsigned numval = shape[slicedim] ; 

    if(NP::VERBOSE) 
    std::cout 
        << " idxx " << DescIdx(idxx)
        << " slicedim " << slicedim 
        << " start " << start 
        << " stride " << stride 
        << " offset " << offset 
        << " numval " << numval 
        << std::endl
        ; 

    const T* vv = cvalues<T>(); 
    out.resize(numval); 
    for(unsigned i=0 ; i < numval ; i++) out[i] = vv[start+i*stride+offset] ; 
}


template<typename T> inline std::string NP::DescSlice(const std::vector<T>& out, unsigned edge )  // static
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < out.size() ; i++ ) 
    {   
         if( i < edge || i > (out.size() - edge) ) 
            ss << std::setw(4) << i << std::setw(15) << std::setprecision(5) << std::fixed << out[i] << std::endl ; 
         else if( i == edge )
            ss << "..." << std::endl; 
    }   
    std::string s = ss.str(); 
    return s ; 
}


template<typename T> inline std::string NP::DescSliceBrief(const std::vector<T>& out )  // static
{
    T mn = std::numeric_limits<T>::max();  
    T mx = std::numeric_limits<T>::min();  

    for(unsigned i=0 ; i < out.size() ; i++ ) 
    {
        T v = out[i] ; 
        if( mn > v ) mn = v ; 
        if( mx < v ) mx = v ; 
    }
    std::stringstream ss ; 
    ss << " mn " << std::setw(15) << std::setprecision(5) << std::fixed << mn ; 
    ss << " mx " << std::setw(15) << std::setprecision(5) << std::fixed << mx ; 
    std::string s = ss.str(); 
    return s ; 
}

inline std::string NP::DescIdx(const std::vector<int>& idxx ) // static
{
    std::stringstream ss ;  
    for(int d=0 ; d < int(idxx.size()) ; d++) ss << idxx[d] << " " ; 
    std::string s = ss.str(); 
    return s ;
}


/**
NP::pickdim__
----------------

Returns ordinal of first -1 in idxx ?

**/

inline int NP::pickdim__(const std::vector<int>& idxx) const
{
    int pd = -1 ; 
    unsigned num = 0 ; 
    for(unsigned d=0 ; d < shape.size() ; d++)  
    {
        int dd = (d < idxx.size() ? idxx[d] : 1) ; 
        if( dd == -1 )
        { 
            if(num == 0) pd = d ; 
            num += 1 ;  
        }
    }
    assert( num == 0 || num == 1 ); 
    return pd ; 
}


/**
NP::index__
-------------

Flat value index ontained from array indices, a -ve index terminates 
the summation over dimensions so only the dimensions to the left of the
-1 are summed.  This is used to from NP::slice to give the start index
of the slice where the slice dimension is marked by the -1.  

**/

inline unsigned NP::index__(const std::vector<int>& idxx) const 
{
    unsigned idx = 0 ; 
    for(unsigned d=0 ; d < shape.size() ; d++)  
    {
        int dd = (d < idxx.size() ? idxx[d] : 1) ; 
        if( dd == -1 ) break ; 
        idx += dd*dimprod(d+1) ;  
    }
    return idx ; 
}


inline unsigned NP::stride__(const std::vector<int>& idxx) const 
{
    int pd = pickdim__(idxx);  
    assert( pd > -1 ); 
    unsigned stride = dimprod(pd+1) ; 
    return stride ; 
}

inline unsigned NP::offset__(const std::vector<int>& idxx) const 
{
    int pd = pickdim__(idxx);  
    assert( pd > -1 ); 

    unsigned offset = 0 ; 
    for(unsigned d=pd+1 ; d < shape.size() ; d++)  
    {
        int dd = (d < idxx.size() ? idxx[d] : 1) ; 
        offset += dd*dimprod(d+1) ;  
    }
    return offset ; 
}






inline unsigned NP::itemsize_(int i, int j, int k, int l, int m, int o) const
{
    return NPS::itemsize_(shape, i, j, k, l, m, o) ; 
}

inline void NP::itembytes_(const char** start,  unsigned& num_bytes,  int i,  int j,  int k,  int l, int m, int o ) const 
{
    unsigned idx0 = index0(i,j,k,l,m,o) ; 
    *start = bytes() + idx0*ebyte ;  

    unsigned sz = itemsize_(i, j, k, l, m, o) ; 
    num_bytes = sz*ebyte ; 
}




template<typename T> inline T NP::get( int i,  int j,  int k,  int l, int m, int o) const 
{
    unsigned idx = index(i, j, k, l, m, o); 
    const T* vv = cvalues<T>() ;  
    return vv[idx] ; 
}

template<typename T> inline void NP::set( T val, int i,  int j,  int k,  int l, int m, int o) 
{
    unsigned idx = index(i, j, k, l, m, o); 
    T* vv = values<T>() ;  
    vv[idx] = val ; 
}



template<typename T> inline bool NP::is_allzero() const 
{
    T zero = T(0) ; 
    const T* vv = cvalues<T>(); 
    int num = 0 ; 
    for(int i=0 ; i < size ; i++) if(vv[i] == zero) num += 1 ; 
    bool allzero = num == size ; 
    return allzero ; 
}

template<typename T> inline void NP::fill(T value)
{
    T* vv = values<T>(); 
    for(int i=0 ; i < size ; i++) *(vv+i) = value ; 
}

template<typename T> inline void NP::_fillIndexFlat(T offset)
{
    T* vv = values<T>(); 
    for(int i=0 ; i < size ; i++) *(vv+i) = T(i) + offset ; 
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
    if(src == nullptr) return nullptr ; 
    NP* dst = new NP(src->dtype); 
    dst->set_shape(src->shape) ; 
    return dst ; 
}

inline void NP::CopyMeta( NP* b, const NP* a ) // static
{
    b->set_shape( a->shape ); 
    b->meta = a->meta ;    // pass along the metadata 
    b->names = a->names ; 
}


inline void NP::set_preserve_last_column_integer_annotation()
{
    set_meta<int>(Preserve_Last_Column_Integer_Annotation, 1 );
}
inline bool NP::is_preserve_last_column_integer_annotation() const 
{
    return 1 == get_meta<int>(Preserve_Last_Column_Integer_Annotation, 0) ; 
}

inline float NP::PreserveNarrowedDoubleInteger( double f )
{
     UIF64 uif64 ; 
     uif64.f = f ; 
     if(VERBOSE) std::cout << "NP::PreserveNarrowedDoubleInteger  uif64.u " << uif64.u << std::endl ; 

     UIF32 uif32 ; 
     uif32.u = int(uif64.u) ; 
     return uif32.f ;  
}

inline NP* NP::MakeNarrow(const NP* a) // static 
{
    assert( a->ebyte == 8 ); 
    std::string b_dtype = NPU::_make_narrow(a->dtype); 

    NP* b = new NP(b_dtype.c_str()); 
    CopyMeta(b, a ); 

    bool plcia = b->is_preserve_last_column_integer_annotation() ; 
    if(VERBOSE && plcia) std::cout 
        << "NP::MakeNarrow"
        << " b.plcia " << plcia 
        << " a.ni " << a->num_items()
        << " b.ni " << b->num_items()
        << " a.iv " << a->num_itemvalues()
        << " b.iv " << b->num_itemvalues()
        << std::endl 
        ; 


    assert( a->num_values() == b->num_values() ); 
    unsigned nv = a->num_values(); 
    unsigned iv = a->num_itemvalues(); 

    if( a->uifc == 'f' && b->uifc == 'f')
    {
        const double* aa = a->cvalues<double>() ;  
        float*        bb = b->values<float>() ;  
        for(unsigned i=0 ; i < nv ; i++) 
        {
            bb[i] = float(aa[i]); 
            bool preserve_last_column_integer = plcia && ((i % iv) == iv - 1 ) ; // only works for 3D not higher D
            if(preserve_last_column_integer) bb[i] = PreserveNarrowedDoubleInteger(aa[i]) ; 
        }
    }

    if(VERBOSE) std::cout 
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
    CopyMeta(b, a ); 

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

    if(VERBOSE) std::cout 
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
    CopyMeta(b, a ); 

    assert( a->arr_bytes() == b->arr_bytes() ); 

    memcpy( b->bytes(), a->bytes(), a->arr_bytes() );    
    unsigned nv = a->num_values(); 

    if(VERBOSE) std::cout 
        << "NP::MakeCopy"
        << " a.dtype " << a->dtype
        << " b.dtype " << b->dtype
        << " nv " << nv
        << std::endl 
        ;

    return b ; 
}

/**
NP::MakeCopy3D
----------------

Copy and change shape to 3D, original dimensions must be 3D or more. 

**/

inline NP* NP::MakeCopy3D(const NP* a) // static 
{
    NP* b = MakeCopy(a); 
    b->change_shape_to_3D(); 
    return b ; 
}

inline NP* NP::ChangeShape3D(NP* a) // static 
{
    a->change_shape_to_3D(); 
    return a ; 
}





inline NP* NP::MakeWideIfNarrow(const NP* a) // static 
{
    return a->ebyte == 4 ? MakeWide(a) : MakeCopy(a) ; 
}
inline NP* NP::MakeNarrowIfWide(const NP* a) // static 
{
    return a->ebyte == 8 ? MakeNarrow(a) : MakeCopy(a) ; 
}

/**
NP::MakeWithType
-------------------

Copies, Narrows or Widens as needed to transform the 
source array into the template type.  
Copies are done when there is no need to narrow or widen 
for memory management consistency.  

**/

template<typename T>
inline NP* NP::MakeWithType(const NP* a) // static 
{
    if(VERBOSE) std::cout 
        << "NP::MakeWithType" 
        << " source type a->ebyte " << a->ebyte 
        << " sizeof(T) " << sizeof(T) 
        << std::endl
        ; 

    assert( sizeof(T) == 4 || sizeof(T) == 8 ); 
    assert( a->ebyte == 4 || a->ebyte == 8 ); 

    NP* b = nullptr ; 
    if( a->ebyte == 4 && sizeof(T) == 4)  
    {
        b = MakeCopy(a); 
    } 
    else if( a->ebyte == 8 && sizeof(T) == 8)
    {
        b = MakeCopy(a); 
    }
    else if( a->ebyte == 8 && sizeof(T) == 4)
    {
        b = MakeNarrow(a) ;  
    }
    else if( a->ebyte == 4 && sizeof(T) == 8)
    {
        b = MakeWide(a) ;  
    }
    return b ; 
}


/**
NP::MakeSelectCopy
--------------------

**/

template<typename... Args> 
inline NP* NP::MakeSelectCopy( const NP* src, Args ... items_ )  // MakeSelectCopy_ellipsis
{
   std::vector<int> items = {items_...};
   return MakeSelectCopy_(src, &items ); 
}

template NP* NP::MakeSelectCopy( const NP* , int ); 
template NP* NP::MakeSelectCopy( const NP* , int, int ); 
template NP* NP::MakeSelectCopy( const NP* , int, int, int ); 
template NP* NP::MakeSelectCopy( const NP* , int, int, int, int ); 

inline NP* NP::MakeSelectCopyE_(  const NP* src, const char* ekey, const char* fallback, char delim )
{
    std::vector<int>* items = U::GetEnvVec<int>(ekey, fallback, delim ); 
    return NP::MakeSelectCopy_( src, items )  ; 
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const char* items_ )
{
    std::vector<int>* items = U::MakeVec<int>(items_); 
    return NP::MakeSelectCopy_( src, items ); 
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const std::vector<int>* items )
{
    return items ? MakeSelectCopy_(src, items->data(), int(items->size()) ) : NP::MakeCopy(src) ; 
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const int* items, int num_items )
{
    assert( items ); 
    for(int i=0 ; i < num_items ; i++) assert( items[i] < int(src->shape[0]) ); 
    std::vector<int> dst_shape(src->shape) ; 
    dst_shape[0] = num_items ; 
    NP* dst = new NP(src->dtype, dst_shape); 
    assert( src->item_bytes() == dst->item_bytes() );  
    unsigned size = src->item_bytes(); 
    for(int i=0 ; i < num_items ; i++) 
    {
        memcpy( dst->bytes() + i*size, src->bytes() + items[i]*size , size ); 
    }
    return dst ; 
}

/**
NP::MakeItemCopy
------------------

Finds the index of a single item from the src array specified by (i,j,k,l,m,n) 
and copies that item into the destination array. 

**/

inline NP* NP::MakeItemCopy(  const NP* src, int i, int j, int k, int l, int m, int o )
{
    std::vector<int> sub_shape ; 
    src->item_shape(sub_shape, i, j, k, l, m, o );   // shape of the item specified by (i,j,k,l,m,n)
    unsigned idx = src->index0(i, j, k, l, m, o ); 

    if(NP::VERBOSE) std::cout 
        << "NP::MakeItemCopy"
        << " i " << i 
        << " j " << j 
        << " k " << k 
        << " l " << l 
        << " m " << m
        << " o " << o
        << " idx " << idx
        << " src.ebyte " << src->ebyte 
        << " src.shape " << NPS::desc(src->shape)
        << " sub_shape " << NPS::desc(sub_shape)
        << std::endl
        ; 

    NP* dst = new NP(src->dtype, sub_shape); 
    memcpy( dst->bytes(), src->bytes() + idx*src->ebyte , dst->arr_bytes() ); 
    return dst ;  
}


/**
NP::item_shape
---------------

Consider an array of the below shape, which has 6 top level items::

   (6, 2, 4096, 4096, 4)

The *item_shape* method returns sub shapes, for example 
a single non-negative argument i=0/1/2/3/4/5 
would yield the the top level items shape:: 

    (2, 4096, 4096, 4 )

Similarly with two non-negative arguments i=0/1/2/3/4/5, j=0/1 
would give item shape:: 

    (4096, 4096, 4 )

**/
inline void NP::item_shape(std::vector<int>& sub, int i, int j, int k, int l, int m, int o ) const 
{
    unsigned nd = shape.size() ; 

    if(i > -1 && j==-1)
    {
        if( nd > 1 ) sub.push_back(shape[1]); 
        if( nd > 2 ) sub.push_back(shape[2]); 
        if( nd > 3 ) sub.push_back(shape[3]); 
        if( nd > 4 ) sub.push_back(shape[4]); 
        if( nd > 5 ) sub.push_back(shape[5]); 
    } 
    else if(i > -1 && j > -1 && k==-1)
    {
        if( nd > 2 ) sub.push_back(shape[2]); 
        if( nd > 3 ) sub.push_back(shape[3]); 
        if( nd > 4 ) sub.push_back(shape[4]); 
        if( nd > 5 ) sub.push_back(shape[5]); 
    }
    else if(i > -1 && j > -1 && k > -1 && l == -1)
    {
        if( nd > 3 ) sub.push_back(shape[3]); 
        if( nd > 4 ) sub.push_back(shape[4]); 
        if( nd > 5 ) sub.push_back(shape[5]); 
    }
    else if(i > -1 && j > -1 && k > -1 && l >  -1 && m == -1)
    {
        if( nd > 4 ) sub.push_back(shape[4]); 
        if( nd > 5 ) sub.push_back(shape[5]); 
    }
    else if(i > -1 && j > -1 && k > -1 && l >  -1 && m > -1 && o == -1)
    {
        if( nd > 5 ) sub.push_back(shape[5]); 
    }
    else if(i > -1 && j > -1 && k > -1 && l >  -1 && m > -1 && o > -1)
    {
        sub.push_back(1); 
    }
}

inline NP* NP::spawn_item(  int i, int j, int k, int l, int m, int o  ) const 
{
    return MakeItemCopy(this, i, j, k, l, m, o ); 
}



/**
NP::MakeCDF
------------

Creating a CDF like this with just plain trapz will usually yield a jerky 
cumulative integral curve. To avoid that need to play some tricks to have 
integral values are more points.

For example by using NP::MakeDiv to split the bins and linearly interpolate
the values. 
 
**/

template<typename T>
inline NP* NP::MakeCDF(const NP* dist )  // static 
{
    NP* cdf = dist->trapz<T>() ;   
    cdf->divide_by_last<T>(); 
    return cdf ; 
}
 

/**
NP::MakeICDF
-------------

Inverts CDF using *nu* NP::pdomain lookups in range 0->1
The input CDF must contain domain and values in the payload last dimension. 
3d or 2d input CDF are accepted where 3d input CDF is interpreted as 
a collection of multiple CDF to be inverted. 

The ICDF created has shape (num_items, nu, hd_factor == 0 ? 1 : 4) 
where num_items is 1 for 2d input CDF and the number of items for 3d input CDF.

Notice that domain information is not included in the output ICDF, this is 
to facilitate direct conversion of the ICDF array into GPU textures.
The *hd_factor* convention regarding domain ranges is used.

Use NP::MakeProperty to add domain infomation using this convention.
 

**/

template<typename T>
inline NP* NP::MakeICDF(const NP* cdf, unsigned nu, unsigned hd_factor, bool dump)  // static 
{
    unsigned ndim = cdf->shape.size(); 
    assert( ndim == 2 || ndim == 3 ); 
    unsigned num_items = ndim == 3 ? cdf->shape[0] : 1 ; 

    assert( hd_factor == 0 || hd_factor == 10 || hd_factor == 20 );  
    T edge = hd_factor > 0 ? T(1.)/T(hd_factor) : 0. ;   

    NP* icdf = new NP(cdf->dtype, num_items, nu, hd_factor == 0 ? 1 : 4 );  
    T* vv = icdf->values<T>(); 

    unsigned ni = icdf->shape[0] ; 
    unsigned nj = icdf->shape[1] ; 
    unsigned nk = icdf->shape[2] ; 

    if(dump) std::cout 
        << "NP::MakeICDF"
        << " nu " << nu
        << " ni " << ni
        << " nj " << nj
        << " nk " << nk
        << " hd_factor " << hd_factor
        << " ndim " << ndim
        << " icdf " << icdf->sstr()
        << std::endl 
        ;

    for(unsigned i=0 ; i < ni ; i++)
    {
        int item = i ;  
        if(dump) std::cout << "NP::MakeICDF" << " item " << item << std::endl ; 

        for(unsigned j=0 ; j < nj ; j++)
        {
            T y_all = T(j)/T(nj) ; //        // 0 -> (nj-1)/nj = 1-1/nj 
            T x_all = cdf->pdomain<T>( y_all, item );    

#ifdef DEBUG
            std::cout 
                <<  " y_all " << std::setw(10) << std::setprecision(4) << std::fixed << y_all 
                <<  " x_all " << std::setw(10) << std::setprecision(4) << std::fixed << x_all 
                << std::endl
                ;
#endif
            unsigned offset = i*nj*nk+j*nk ;  

            vv[offset+0] = x_all ;

            if( hd_factor > 0 )
            {
                T y_lhs = T(j)/T(hd_factor*nj) ;
                T y_rhs = T(1.) - edge + T(j)/T(hd_factor*nj) ; 

                T x_lhs = cdf->pdomain<T>( y_lhs, item );    
                T x_rhs = cdf->pdomain<T>( y_rhs, item );    

                vv[offset+1] = x_lhs ;
                vv[offset+2] = x_rhs ;
                vv[offset+3] = 0. ;
            }
        }
    }
    return icdf ; 
} 

/**
NP::MakeProperty
-----------------

For hd_factor=0 converts a one dimensional array of values with shape (ni,)
into 2d array of shape (ni, 2) with the domain a range of values 
from 0 -> (ni-1)/ni = 1-1/ni 
Thinking in one dimensional terms that means that values and 
corresponding domains get interleaved.
The resulting property array can then be used with NP::pdomain or NP::interp.

For hd_factor=10 or hd_factor=20 the input array is required to have shape (ni,4) or (ni,nj,4)
where "all" is in payload slot 0 and lhs and rhs high resolution zooms are in 
payload slots 1 and 2.  (Slot 3 is currently spare, normally containing zero). 

The output array has an added dimension with shape  (ni,4,2) 
adding domain values interleaved with the values. 
The domain values follow the hd_factor convention of scaling the resolution 
in the 1/hd_factor tails


**/

template <typename T> NP* NP::MakeProperty(const NP* a, unsigned hd_factor ) // static 
{
    NP* prop = nullptr ; 
    unsigned ndim = a->shape.size(); 
    assert( ndim == 1 || ndim == 2 || ndim == 3 ); 

    if( ndim == 1 )
    {
        assert( hd_factor == 0 );  

        unsigned ni = a->shape[0] ; 
        unsigned nj = 2 ; 
        prop = NP::Make<T>(ni, nj) ; 
        T* prop_v = prop->values<T>(); 
        for(unsigned i=0 ; i < ni ; i++)
        {
            prop_v[nj*i+0] = T(i)/T(ni) ;  // 0 -> (ni-1)/ni = 1-1/ni 
            prop_v[nj*i+1] = a->get<T>(i) ; 
        }
    } 
    else if( ndim == 2 )   
    {
        assert( hd_factor == 10 || hd_factor == 20 ); 
        T edge = 1./T(hd_factor) ;
        unsigned ni = a->shape[0] ; 
        unsigned nj = a->shape[1] ; assert( nj == 4 ); 
        unsigned nk = 2 ; 

        prop = NP::Make<T>(ni, nj, nk) ; 
        T* prop_v = prop->values<T>(); 

        for(unsigned i=0 ; i < ni ; i++)
        {
            T u_all =  T(i)/T(ni) ; 
            T u_lhs =  T(i)/T(hd_factor*ni) ; 
            T u_rhs =  1. - edge + T(i)/T(hd_factor*ni) ; 
            T u_spa =  0. ; 

            for(unsigned j=0 ; j < nj ; j++)   // 0,1,2,3
            {
                unsigned k;
                k=0 ; 
                switch(j)
                {
                    case 0:prop_v[nk*nj*i+nk*j+k] = u_all ; break ; 
                    case 1:prop_v[nk*nj*i+nk*j+k] = u_lhs ; break ; 
                    case 2:prop_v[nk*nj*i+nk*j+k] = u_rhs ; break ; 
                    case 3:prop_v[nk*nj*i+nk*j+k] = u_spa ; break ; 
                }
                k=1 ;  
                prop_v[nk*nj*i+nk*j+k] = a->get<T>(i,j) ; 
            }
        }
    }
    else if( ndim == 3 )
    {
        assert( hd_factor == 10 || hd_factor == 20 ); 
        T edge = 1./T(hd_factor) ;
        unsigned ni = a->shape[0] ; 
        unsigned nj = a->shape[1] ; 
        unsigned nk = a->shape[2] ; assert( nk == 4 );   // hd_factor convention
        unsigned nl = 2 ; 

        prop = NP::Make<T>(ni, nj, nk, nl) ; 

        for(unsigned i=0 ; i < ni ; i++)
        {
            for(unsigned j=0 ; j < nj ; j++)
            {
                T u_all =  T(j)/T(nj) ; 
                T u_lhs =  T(j)/T(hd_factor*nj) ; 
                T u_rhs =  1. - edge + T(j)/T(hd_factor*nj) ; 
                T u_spa =  0. ; 

                for(unsigned k=0 ; k < nk ; k++)   // 0,1,2,3
                {
                    unsigned l ; 
                    l=0 ; 
                    switch(k)
                    {
                        case 0:prop->set<T>(u_all, i,j,k,l) ; break ; 
                        case 1:prop->set<T>(u_lhs, i,j,k,l) ; break ; 
                        case 2:prop->set<T>(u_rhs, i,j,k,l) ; break ; 
                        case 3:prop->set<T>(u_spa, i,j,k,l) ; break ; 
                    }
                    l=1 ;  
                    prop->set<T>( a->get<T>(i,j,k), i,j,k,l );   
                }
            }
        }
    }
    return prop ; 
}

/**
NP::MakeLookupSample
-----------------------

Create a lookup sample of shape (ni,) using the 2d icdf_prop and ni uniform random numbers 
Hmm in regions where the CDF is flat (and ICDF is steep), the ICDF lookup does not do very well.
That is the reason for hd_factor, to increase resolution at the extremes where this 
issue usually occurs without paying the cost of higher resolution across the entire range.

TODO: compare what this provides directly on the ICDF (using NP::interp) 
      with what the CDF directly can provide (using NP::pdomain)
     
**/

template <typename T> NP* NP::MakeLookupSample(const NP* icdf_prop, unsigned ni, unsigned seed, unsigned hd_factor ) // static 
{
    unsigned ndim = icdf_prop->shape.size() ; 
    unsigned npay = icdf_prop->shape[ndim-1] ; 
    assert( npay == 2 ); 

    if(ndim == 2)
    {
        assert( hd_factor == 0 ); 
    }
    else if( ndim == 3 )
    {
        assert( hd_factor == 10 || hd_factor == 20  ); 
        assert( icdf_prop->shape[1] == 4 ); 
    }

    std::mt19937_64 rng;
    rng.seed(seed); 
    std::uniform_real_distribution<T> unif(0, 1);

    NP* sample = NP::Make<T>(ni); 
    T* sample_v = sample->values<T>(); 
    for(unsigned i=0 ; i < ni ; i++) 
    {
        T u = unif(rng) ;  
        T y = hd_factor > 0 ? icdf_prop->interpHD<T>(u, hd_factor ) : icdf_prop->interp<T>(u) ; 
        sample_v[i] = y ; 
    }
    return sample ; 
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










inline NP* NP::copy() const 
{
    return MakeCopy(this); 
}


inline NP* NP::LoadWide(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ; 
    std::string path = form_path(dir, reldir, name); 
    return LoadWide(path.c_str());
}

inline NP* NP::LoadWide(const char* dir, const char* name)
{
    if(!dir) return nullptr ; 
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
    if(!path) return nullptr ; 
    NP* a = NP::Load(path);  

    assert( a->uifc == 'f' && ( a->ebyte == 8 || a->ebyte == 4 ));  
    // cannot think of application for doing this with  ints, so restrict to float OR double 

    NP* b = a->ebyte == 8 ? NP::MakeCopy(a) : NP::MakeWide(a) ;  

    a->clear(); 

    return b ; 
}


inline NP* NP::LoadNarrow(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ; 
    std::string path = form_path(dir, reldir, name); 
    return LoadNarrow(path.c_str());
}

inline NP* NP::LoadNarrow(const char* dir, const char* name)
{
    if(!dir) return nullptr ; 
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
    if(!path) return nullptr ; 
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
    else if( ibin == int(ni) )
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



template<typename T> inline void NP::pdump(const char* msg, T d_scale, T v_scale) const  
{
    bool property_shaped = is_pshaped(); 
    assert( property_shaped ); 

    unsigned ni = shape[0] ; 
    std::cout 
        << msg 
        << " ni " << ni 
        << " d_scale " 
        << std::fixed << std::setw(10) << std::setprecision(5) << d_scale
        << " v_scale " 
        << std::fixed << std::setw(10) << std::setprecision(5) << v_scale
        << std::endl
        ; 

    const T* vv = cvalues<T>(); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        std::cout 
             << " i " << std::setw(3) << i 
             << " d " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+0]*d_scale 
             << " v " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+1]*v_scale 
             << std::endl
             ; 
    }
}

/**
NP::minmax
------------

Finds minimum and maximum values of column j, assuming a 2d array, 
by looping over the first array dimension and comparing all values. 

**/

template<typename T> inline void NP::minmax(T& mn, T&mx, unsigned j, int item ) const 
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 2 || ndim == 3);  

    unsigned ni = shape[ndim-2] ; 
    unsigned nj = shape[ndim-1] ; 
    assert( j < nj ); 

    unsigned num_items = ndim == 3 ? shape[0] : 1 ; 
    assert( item < int(num_items) ); 
    unsigned item_offset = item == -1 ? 0 : ni*nj*item ; 
    const T* vv = cvalues<T>() + item_offset ;  // shortcut approach to handling multiple items 

    mn = std::numeric_limits<T>::max() ; 
    mx = std::numeric_limits<T>::min() ; 
 
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

template<typename T> inline NP* NP::trapz() const 
{
    assert( shape.size() == 2 && shape[1] == 2 && shape[0] > 1); 
    unsigned ni = shape[0] ; 
    T half(0.5); 
    T xmn = get<T>(0, 0); 

    NP* integral = NP::MakeLike(this); 
    T* integral_v = integral->values<T>(); 
    integral_v[0] = xmn ; 
    integral_v[1] = 0. ; 

    for(unsigned i=0 ; i < ni-1 ; i++)
    {
        T x0 = get<T>(i, 0);
        T y0 = get<T>(i, 1);

        T x1 = get<T>(i+1, 0); 
        T y1 = get<T>(i+1, 1); 

#ifdef DEBUG
        std::cout 
            << " x0 " << std::setw(10) << std::fixed << std::setprecision(4) << x0 
            << " y0 " << std::setw(10) << std::fixed << std::setprecision(4) << y0
            << " x1 " << std::setw(10) << std::fixed << std::setprecision(4) << x1
            << " y1 " << std::setw(10) << std::fixed << std::setprecision(4) << y1
            << std::endl 
            ;
#endif
        integral_v[2*(i+1)+0] = x1 ;  // x0 of first bin covered with xmn
        integral_v[2*(i+1)+1] = integral_v[2*(i+0)+1] + (x1 - x0)*(y0 + y1)*half ;  
    } 
    return integral ;  
}

template<typename T> void NP::psplit(std::vector<T>& dom, std::vector<T>& val) const 
{
    unsigned nv = num_values() ; 
    const T* vv = cvalues<T>() ; 

    assert( nv %  2 == 0 );  
    unsigned entries = nv/2 ;

    dom.resize(entries); 
    val.resize(entries); 

    for(unsigned i=0 ; i < entries ; i++)
    {   
        dom[i] = vv[2*i+0] ; 
        val[i] = vv[2*i+1] ; 
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

    assert( nj <= 8 );        // not needed for below, just for sanity of payload
    unsigned jdom = 0 ;       // 1st payload slot is "domain"
    unsigned jval = nj - 1 ;  // last payload slot is "value" 
    // note that with nj > 2 this allows other values to be carried 

    unsigned num_items = ndim == 3 ? shape[0] : 1 ; 
    assert( item < int(num_items) ); 
    unsigned item_offset = item == -1 ? 0 : ni*nj*item ;   // using item = 0 will have the same effect

    const T* vv = cvalues<T>() + item_offset ;  // shortcut approach to handling multiple items 


    const T lhs_dom = vv[nj*(0)+jdom]; 
    const T rhs_dom = vv[nj*(ni-1)+jdom];
    bool dom_expect = rhs_dom >= lhs_dom  ;  // allow equal as getting zeros at extremes 

    const T lhs_val = vv[nj*(0)+jval]; 
    const T rhs_val = vv[nj*(ni-1)+jval];
    bool val_expect = rhs_val >= lhs_val ; 

    if(!dom_expect) std::cout 
        << "NP::pdomain FATAL dom_expect : rhs_dom > lhs_dom "
        << " lhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_dom 
        << " rhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_dom 
        << std::endl 
        ;
    assert( dom_expect ); 

    if(!val_expect) std::cout 
        << "NP::pdomain FATAL val_expect : rhs_val > lhs_val "
        << " lhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_val 
        << " rhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_val 
        << std::endl 
        ;
    assert( val_expect ); 

    const T yv = value ; 
    T xv ;   
    bool xv_set = false ; 


    if( yv <= lhs_val )
    {
        xv = lhs_dom ; 
        xv_set = true ; 
    }
    else if( yv >= rhs_val )
    {
        xv = rhs_dom  ; 
        xv_set = true ; 
    }
    else if ( yv >= lhs_val && yv < rhs_val  )
    {
        for(unsigned i=0 ; i < ni-1 ; i++) 
        {
            const T x0 = vv[nj*(i+0)+jdom] ; 
            const T y0 = vv[nj*(i+0)+jval] ; 
            const T x1 = vv[nj*(i+1)+jdom] ; 
            const T y1 = vv[nj*(i+1)+jval] ;
            const T dy = y1 - y0 ;  

            //assert( dy >= zero );   // must be monotonic for this to make sense
            /*
            if( dy < zero )
            {  
                std::cout 
                    << "NP::pdomain ERROR : non-monotonic dy less than zero  " 
                    << " i " << std::setw(5) << i
                    << " x0 " << std::setw(10) << std::fixed << std::setprecision(6) << x0
                    << " x1 " << std::setw(10) << std::fixed << std::setprecision(6) << x1 
                    << " y0 " << std::setw(10) << std::fixed << std::setprecision(6) << y0
                    << " y1 " << std::setw(10) << std::fixed << std::setprecision(6) << y1 
                    << " yv " << std::setw(10) << std::fixed << std::setprecision(6) << yv
                    << " dy " << std::setw(10) << std::fixed << std::setprecision(6) << dy 
                    << std::endl 
                    ;
            }
            */

            if( y0 <= yv && yv < y1 )
            { 
                xv = x0 ; 
                xv_set = true ; 
                if( dy > zero ) xv += (yv-y0)*(x1-x0)/dy ; 
                break ;   
            }
        }
    } 

    assert( xv_set == true ); 

    if(dump)
    {
        std::cout 
            << "NP::pdomain.dump "
            << " item " << std::setw(4) << item
            << " ni " << std::setw(4) << ni
            << " nj " << std::setw(4) << nj
            << " lhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_dom
            << " rhs_dom " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_dom
            << " lhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << lhs_val
            << " rhs_val " << std::setw(10) << std::fixed << std::setprecision(4) << rhs_val
            << " yv " << std::setw(10) << std::fixed << std::setprecision(4) << yv
            << " xv " << std::setw(10) << std::fixed << std::setprecision(4) << xv
            << std::endl 
            ; 
    }
    return xv ; 
}


/**
NP::interp2D
-------------

* https://en.wikipedia.org/wiki/Bilinear_interpolation

The interpolation formulas used by CUDA textures are documented.

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering

::

    J.2. Linear Filtering
    In this filtering mode, which is only available for floating-point textures, the value returned by the texture fetch is

    tex(x)=(1−α)T[i]+αT[i+1] for a one-dimensional texture,

    tex(x,y)=(1−α)(1−β)T[i,j]+α(1−β)T[i+1,j]+(1−α)βT[i,j+1]+αβT[i+1,j+1] for a two-dimensional texture,

    tex(x,y,z) =
    (1−α)(1−β)(1−γ)T[i,j,k]+α(1−β)(1−γ)T[i+1,j,k]+
    (1−α)β(1−γ)T[i,j+1,k]+αβ(1−γ)T[i+1,j+1,k]+
    (1−α)(1−β)γT[i,j,k+1]+α(1−β)γT[i+1,j,k+1]+
    (1−α)βγT[i,j+1,k+1]+αβγT[i+1,j+1,k+1]

    for a three-dimensional texture,
    where:

    i=floor(xB), α=frac(xB), xB=x-0.5,
    j=floor(yB), β=frac(yB), yB=y-0.5,
    k=floor(zB), γ=frac(zB), zB= z-0.5,
    α, β, and γ are stored in 9-bit fixed point format with 8 bits of fractional value (so 1.0 is exactly represented).


The use of reduced precision makes it not straightforward to perfectly replicate on the CPU, 
but you should be able to get very close. 


**/


/**
NP::interp
------------

CAUTION: using the wrong type here somehow scrambles the array contents, 
so always explicitly define the template type : DO NOT RELY ON COMPILER WORKING IT OUT.

**/

template<typename T> inline T NP::interp(T x, int item) const  
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 2 || ndim == 3 ); 

    unsigned num_items = ndim == 3 ? shape[0] : 1 ; 
    assert( item < int(num_items) ); 
    unsigned ni = shape[ndim-2]; 
    unsigned nj = shape[ndim-1];  // typically 2 but can be more
    unsigned item_offset = item == -1 ? 0 : ni*nj*item ;   // item=-1 same as item=0

    assert( ni > 1 ); 
    assert( nj <= 8 );        // not needed for below, just for sanity of payload
    unsigned jdom = 0 ;       // 1st payload slot is "domain"
    unsigned jval = nj - 1 ;  // last payload slot is "value"   : TODO: argument to control this
    // note that with nj > 2 this allows other values to be carried 

    const T* vv = cvalues<T>() + item_offset ; 

    int lo = 0 ;
    int hi = ni-1 ;         // domain must be in ascending order 

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

    // for x out of domain range return values at edges
    if( x <= vv[nj*lo+jdom] ) return vv[nj*lo+jval] ; 
    if( x >= vv[nj*hi+jdom] ) return vv[nj*hi+jval] ; 

    // binary search for domain bin containing x 
    while (lo < hi-1)
    {
        int mi = (lo+hi)/2;
        if (x < vv[nj*mi+jdom]) hi = mi ;
        else lo = mi;
    }

    // linear interpolation across the bin 
    T dx = vv[nj*hi+jdom] - vv[nj*lo+jdom] ; 
    T fx = (x-vv[nj*lo+jdom])/dx ; 

    // "hi = lo + 1", so  fractional "lo + fx" 
    // encodes the result of the domain interpolation 
    // HMM need some signalling for below/above domain
    // 
    // Notice how the values are only used in the below two lines, right at the end.
    // Could split the API to return a fractional domain bin "index".
    // 
    // Then could reuse that to get multiple values from a single "bin_interp"  
    // Could interpolate multiple props without repeating the bin finding so long as
    // they shared the same domain (in first column)

    T dy = vv[nj*hi+jval] - vv[nj*lo+jval] ; 
    T y  = vv[nj*lo+jval] + dy*fx ; 

    return y ; 
}

/**
NP::interpHD
--------------

Interpolation within domain 0->1 using hd_factor convention for lhs, rhs high resolution zooms. 

Previously tried to avoid the dimensional duplication using set_offset 
which attempts to enable get/set addressing with the "wrong" number of dimensions.
The offset is like moving a cursor around the array allowing portions of it 
to be in-situ addressed as if they were smaller sub-arrays. 

The set_offset approach is problematic as the get/set methods are using the 
absolute "correct" ni,nj,nk etc.. whereas when using the lower level cvalues approach 
are able to shift the meanings of those in a local fashion that can work 
across different numbers of dimensions.

**/

template<typename T> inline T NP::interpHD(T u, unsigned hd_factor, int item) const 
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 3 || ndim == 4 ); 

    unsigned num_items = ndim == 4 ? shape[0] : 1 ; 
    assert( item < int(num_items) ); 

    unsigned ni = shape[ndim-3] ; 
    unsigned nj = shape[ndim-2] ; 
    unsigned nk = shape[ndim-1] ; 
    assert( nj == 4 );
    assert( nk == 2 ); // not required by the below 

    unsigned kdom = 0 ; 
    unsigned kval = nk - 1 ; 

    // pick *j* resolution zoom depending on u 
    T lhs = T(1.)/T(hd_factor) ; 
    T rhs = T(1.) - lhs ; 
    unsigned j = u > lhs && u < rhs ? 0 : ( u < lhs ? 1 : 2 ) ;  

    unsigned item_offset = item == -1 ? 0 : ni*nj*nk*item ;   // item=-1 same as item=0
    const T* vv = cvalues<T>() + item_offset ; 

    // lo and hi are standins for *i*
    int lo = 0 ;
    int hi = ni-1 ;

    if( u <= vv[lo*nj*nk+j*nk+kdom] ) return vv[lo*nj*nk+j*nk+kval] ; 
    if( u >= vv[hi*nj*nk+j*nk+kdom] ) return vv[hi*nj*nk+j*nk+kval] ; 

    // binary search for domain bin containing x 
    while (lo < hi-1)
    {
        int mi = (lo+hi)/2;
        if (u < vv[mi*nj*nk+j*nk+kdom] ) hi = mi ;
        else lo = mi;
    }

    // linear interpolation across the bin 
    T dy = vv[hi*nj*nk+j*nk+kval] - vv[lo*nj*nk+j*nk+kval] ; 
    T du = vv[hi*nj*nk+j*nk+kdom] - vv[lo*nj*nk+j*nk+kdom] ; 
    T y  = vv[lo*nj*nk+j*nk+kval] + dy*(u-vv[lo*nj*nk+j*nk+kdom])/du ; 

    return y ; 
}


/**
NP::interp 
-------------

Too dangerous to simply remove this method, as the standard NP::interp 
has too similar a signature which via type conversion could lead to 
difficult to find bugs. 

**/

template<typename T> inline T NP::interp(int i, T x) const  
{  
    std::cerr << "NP::interp DEPRECATED SIGNATURE CHANGE NP::interp TO NP::combined_interp " << std::endl ; 
    return combined_interp_3<T>(i, x ); 
}

/**
NP::combined_interp_3
------------------------

Assuming a convention of combined property array layout 
this provides interpolation of multiple properties with 
different domain lengths.  Special array preparation 
is needed with "ni" lengths encoded into last columns, for 
example with NP::Combine

See ~/np/tests/NPCombineTest.cc

qudarap/qprop.h qprop<T>::interpolate
    GPU version of NP::combined_interp provisioned by qudarap/QProp.hh


::

    In [1]: a.shape
    Out[1]: (24, 15, 2)

    In [2]: a[:,-1,-1]
    Out[2]: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    In [3]: a[:,-1,-1].view(np.int64)
    Out[3]: array([10,  2, 14,  2, 14, 14,  4,  2, 10,  2, 14,  2, 14, 14,  4,  2, 10,  2, 14,  2, 14, 14,  4,  2])


    In [18]: a.shape
    Out[18]: (3, 4, 2, 15, 2)

    In [17]: a[...,-1,-1].view(np.int64)
    Out[17]: 
    array([[[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]],

           [[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]],

           [[10,  2],
            [14,  2],
            [14, 14],
            [ 4,  2]]])


    std::cout 
         << " NP::combined_interp_3 "
         << " x " << x 
         << " ni " << ni 
         << " lo " << lo
         << " hi " << hi
         << " vx_lo " <<  vv[nj*lo+jdom] 
         << " vy_lo " <<  vv[nj*lo+jval] 
         << " vx_hi " <<  vv[nj*hi+jdom] 
         << " vy_hi " <<  vv[nj*hi+jval] 
         << std::endl
         ; 

**/

template<typename T> inline T NP::combined_interp_3(int i, T x) const  
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 3 && shape[ndim-1] >= 2 && i < shape[0] && shape[1] > 1 ); 

    unsigned stride = shape[ndim-2]*shape[ndim-1] ; 
    const T* vv = cvalues<T>() + i*stride ; 

    return _combined_interp<T>( vv, stride, x ); 
}


/**
NP::combined_interp_5
--------------------------

Example array layout for complex refractive index::

      (3 pmtcat, 4 layers, 2 prop ,  ~15  ,  2 )   
                           |                 |
                           RINDEX  1+mx_itm  dom
                           KINDEX            val  

**/

template<typename T> inline T NP::combined_interp_5(int i, int j, int k, T x) const  
{
    unsigned ndim = shape.size() ; 
    assert( ndim == 5 ); 
    unsigned ni = shape[0] ; 
    unsigned nj = shape[1] ; 
    unsigned nk = shape[2] ; 
    assert( i < ni && j < nj && k < nk ); 

    unsigned nl = shape[ndim-2] ; 
    unsigned nm = shape[ndim-1] ; 
    assert( nl > 1 );   // require more than one domain items  
    assert( nm == 2 ); 

    unsigned stride = shape[ndim-2]*shape[ndim-1] ; 
    unsigned iprop = i*nj*nk+j*nk+k ;   
    // maximum:  (ni-1)*nj*nk + (nj-1)*nk + (nk-1) = ni*nj*nk - nj*nk + nj*nk - nk + nk - 1 = ni*nj*nk - 1 
    
    const T* vv = cvalues<T>() + iprop*stride ; 

    return _combined_interp<T>( vv, stride, x ); 
}


/**
NP::_combined_interp
----------------------

Note how this needs not know about the upper dimensions, allowing the split 

Using ragged array handling with NP::Combined arrays 
where individual property (ni,2) have the ni encoded into the absolute 
last column. 

**/

template<typename T> inline T NP::_combined_interp(const T* vv, unsigned niv, T x) const  
{
    unsigned ndim = shape.size() ; 
    unsigned ni = nview::int_from<T>( *(vv+niv-1) ) ; // NPU.hh:nview 
    unsigned nj = shape[ndim-1] ;  // normally 2 with (dom, val)

    unsigned jdom = 0 ;       // 1st payload slot is "domain"
    unsigned jval = nj - 1 ;  // last payload slot is "value", normally 1  

    int lo = 0 ;
    int hi = ni-1 ;

    if( x <= vv[nj*lo+jdom] ) return vv[nj*lo+jval] ; 
    if( x >= vv[nj*hi+jdom] ) return vv[nj*hi+jval] ; 

    while (lo < hi-1)
    {
        int mi = (lo+hi)/2;
        if (x < vv[nj*mi+jdom]) hi = mi ;
        else lo = mi;
    }

    T dy = vv[nj*hi+jval] - vv[nj*lo+jval] ; 
    T dx = vv[nj*hi+jdom] - vv[nj*lo+jdom] ; 
    T y =  vv[nj*lo+jval] + dy*(x-vv[nj*lo+jdom])/dx ; 
    return y ; 
}




template<typename T> inline NP* NP::cumsum(int axis) const  
{
    assert( axis == 1 && "for now only axis=1 implemented" ); 
    const T* vv = cvalues<T>(); 
    NP* cs = NP::MakeLike(this) ; 
    T* ss = cs->values<T>(); 
    for(int p=0 ; p < size ; p++) ss[p] = vv[p] ;   // flat copy 

    unsigned ndim = shape.size() ; 

    if( ndim == 1 )
    {
        unsigned ni = shape[0] ; 
        for(unsigned i=1 ; i < ni ; i++) ss[i] += ss[i-1] ;   // cute recursive summation
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

Normalization by last payload entry implemented for 1d, 2d and 3d arrays.

**/

template<typename T> inline void NP::divide_by_last() 
{
    unsigned ndim = shape.size() ; 
    T* vv = values<T>(); 
    const T zero(0.); 

    if( ndim == 1 )
    {
        unsigned ni = shape[0] ; 
        const T last = get<T>(-1) ; 
        for(unsigned i=0 ; i < ni ; i++) vv[i] /= last  ;  
    }
    else if( ndim == 2 )
    {
        unsigned ni = shape[0] ; 
        unsigned nj = shape[1] ; 
#ifdef DEBUG
        std::cout 
            << "NP::divide_by_last 2d "
            << " ni " << ni  
            << " nj " << nj
            << std::endl
            ;  
#endif
        // 2d case ni*(domain,value) pairs : there is only one last value to divide by : like the below 3d case with ni=1, i=0 
        const T last = get<T>(-1,-1) ;
        unsigned j = nj - 1 ;    // last payload slot    
        for(unsigned i=0 ; i < ni ; i++)
        {
            if(last != zero) vv[i*nj+j] /= last ;  
        }
    }
    else if( ndim == 3 )   // eg (1000, 100, 2)    1000(diff BetaInverse) * 100 * (energy, integral)  
    {
        unsigned ni = shape[0] ;  // eg BetaInverse dimension
        unsigned nj = shape[1] ;  // eg energy dimension 
        unsigned nk = shape[2] ;  // eg payload carrying  [energy,s2,s2integral]
        assert( nk <= 8  ) ;      // not required by the below, but restrict for understanding 
        unsigned k = nk - 1 ;     // last payload property, eg s2integral

        for(unsigned i=0 ; i < ni ; i++)
        {
            // get<T>(i, -1, -1 )
            const T last = vv[i*nj*nk+(nj-1)*nk+k] ;  // for each item i, pluck the last payload value at the last energy value 
            for(unsigned j=0 ; j < nj ; j++) if(last != zero) vv[i*nj*nk+j*nk+k] /= last ;  // traverse energy dimension normalizing the last payload items by last energy brethren
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


inline void NP::dump(int i0, int i1, int j0, int j1) const 
{
    if(uifc == 'f')
    {   
        switch(ebyte)
        {   
            case 4: _dump<float>(i0,i1,j0,j1)  ; break ; 
            case 8: _dump<double>(i0,i1,j0,j1) ; break ; 
        }   
    }   
    else if(uifc == 'u')
    {   
        switch(ebyte)
        {   
            case 1: _dump<unsigned char>(i0,i1,j0,j1)  ; break ; 
            case 2: _dump<unsigned short>(i0,i1,j0,j1)  ; break ; 
            case 4: _dump<unsigned int>(i0,i1,j0,j1) ; break ; 
            case 8: _dump<unsigned long>(i0,i1,j0,j1) ; break ; 
        }   
    }   
    else if(uifc == 'i')
    {   
        switch(ebyte)
        {   
            case 1: _dump<char>(i0,i1,j0,j1)  ; break ; 
            case 2: _dump<short>(i0,i1,j0,j1)  ; break ; 
            case 4: _dump<int>(i0,i1,j0,j1) ; break ; 
            case 8: _dump<long>(i0,i1,j0,j1) ; break ; 
        }   
    }   
}

inline std::string NP::sstr() const 
{
    std::stringstream ss ; 
    ss << NPS::desc(shape) ; 
    return ss.str(); 
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
       << " names.size " << names.size() 
       ;
    return ss.str(); 
}

inline std::string NP::brief() const 
{
    std::stringstream ss ; 
    ss 
       << " " << dtype
       << NPS::desc(shape) 
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





inline void NP::set_names( const std::vector<std::string>& lines )
{
    names.clear(); 
    for(unsigned i=0 ; i < lines.size() ; i++)
    {
         const std::string& line = lines[i] ; 
         names.push_back(line); 
    }
}

inline void NP::get_names( std::vector<std::string>& lines ) const 
{
    for(unsigned i=0 ; i < names.size() ; i++)
    {
         const std::string& name = names[i] ; 
         lines.push_back(name); 
    }
}

//Returns 0-based index of first matching name, or -1 if the name is not found or the name is nullptr. 
inline int NP::get_name_index( const char* qname ) const 
{
    unsigned count = 0 ; 
    return NameIndex(qname, count, names); 
}
inline int NP::get_name_index( const char* qname, unsigned& count ) const 
{
    return NameIndex(qname, count, names); 
}


/**
NP::NameIndex
--------------------

Returns the index of the first listed name that exactly matches the query string.
A count of the number of matches is also provided.
Returns -1 if not found.

**/

inline int NP::NameIndex( const char* qname, unsigned& count, const std::vector<std::string>& names ) // static
{
    if(names.size() == 0) return -1 ; 

    int result(-1); 
    count = 0 ; 
    for(unsigned i=0 ; i < names.size() ; i++)
    {   
        const std::string& k = names[i] ;
        if(strcmp(k.c_str(), qname) == 0 ) 
        {   
            if(count == 0) result = i ; 
            count += 1 ;   
        }   
    }   
    return result ; 
}


inline bool NP::is_named_shape() const 
{
    return int(shape.size()) == 2 && shape[1] == 1 && shape[0] == int(names.size()) ; 
}

template<typename T>
inline T NP::get_named_value( const char* qname, T fallback ) const 
{
    bool is_named = is_named_shape() ; 
    if(! is_named) return fallback ; 

    const T* vv = cvalues<T>() ; 
  
    unsigned count(0); 
    int idx = get_name_index(qname, count ); 

    if(count != 1) return fallback ; 
    if(idx < int(shape[0])) return vv[idx] ; 
    return fallback ; 
}






/**
NP::get_meta_string_
----------------------

Assumes metadata layout of form::

    key1:value1
    key2:value2

With each key-value pair separated by newlines and the key and value
delimited by a colon.

**/

inline std::string NP::get_meta_string_(const char* metadata, const char* key) // static 
{
    std::string value ; 

    std::stringstream ss;
    ss.str(metadata);
    std::string s;
    char delim = ':' ; 

    while (std::getline(ss, s))
    { 
       size_t pos = s.find(delim); 
       if( pos != std::string::npos )
       {
           std::string k = s.substr(0, pos);
           std::string v = s.substr(pos+1);
           if(strcmp(k.c_str(), key) == 0 ) value = v ; 
#ifdef DEBUG
           std::cout 
               << "NP::get_meta_string " 
               << " s[" << s << "]"
               << " k[" << k << "]" 
               << " v[" << v << "]" 
               << std::endl
               ;   
#endif
       }
#ifdef DEBUG
       else
       {
           std::cout 
               << "NP::get_meta_string " 
               << "s[" << s << "] SKIP "   
               << std::endl
               ;   
       }
#endif
    } 
    return value ; 
}

inline std::string NP::get_meta_string(const std::string& meta, const char* key) 
{
    const char* metadata = meta.empty() ? nullptr : meta.c_str() ; 
    return get_meta_string_( metadata, key ); 
}

template<typename T> inline T NP::GetMeta(const std::string& mt, const char* key, T fallback) // static 
{
    if(mt.empty()) return fallback ; 
    std::string s = get_meta_string( mt, key); 
#ifdef DEBUG
    std::cout << "NP::GetMeta[" << s << "]" << std::endl ;
#endif
    if(s.empty()) return fallback ; 
    return To<T>(s.c_str()) ; 
}


template int         NP::GetMeta<int>(        const std::string& , const char*, int ) ; 
template unsigned    NP::GetMeta<unsigned>(   const std::string& , const char*, unsigned ) ; 
template float       NP::GetMeta<float>(      const std::string& , const char*, float ) ; 
template double      NP::GetMeta<double>(     const std::string& , const char*, double ) ; 
template std::string NP::GetMeta<std::string>(const std::string& , const char*, std::string ) ; 


inline bool NP::has_meta() const 
{
    return meta.empty() == false ; 
}

template<typename T> inline T NP::get_meta(const char* key, T fallback) const 
{
    if(meta.empty()) return fallback ; 
    return GetMeta<T>( meta.c_str(), key, fallback ); 
}

template int      NP::get_meta<int>(const char*, int ) const ; 
template unsigned NP::get_meta<unsigned>(const char*, unsigned ) const  ; 
template float    NP::get_meta<float>(const char*, float ) const ; 
template double   NP::get_meta<double>(const char*, double ) const ; 
template std::string NP::get_meta<std::string>(const char*, std::string ) const ; 



template<typename T> inline void NP::SetMeta( std::string& mt, const char* key, T value ) // static
{
    std::stringstream nn;
    std::stringstream ss;
    ss.str(mt);
    std::string s;
    char delim = ':' ; 
    bool changed = false ; 
    while (std::getline(ss, s))
    { 
       size_t pos = s.find(delim); 
       if( pos != std::string::npos )
       {
           std::string k = s.substr(0, pos);
           std::string v = s.substr(pos+1);
           if(strcmp(k.c_str(), key) == 0 )  // key already present, so change it 
           {
               changed = true ; 
               nn << key << delim << value << std::endl ;   
           }
           else
           {
               nn << s << std::endl ;  
           }
       }
       else
       {
           nn << s << std::endl ;  
       }    
    }
    if(!changed) nn << key << delim << value << std::endl ; 
    mt = nn.str() ; 
}

template void     NP::SetMeta<int>(         std::string&, const char*, int ); 
template void     NP::SetMeta<unsigned>(    std::string&, const char*, unsigned ); 
template void     NP::SetMeta<float>(       std::string&, const char*, float ); 
template void     NP::SetMeta<double>(      std::string&, const char*, double ); 
template void     NP::SetMeta<std::string>( std::string&, const char*, std::string ); 


inline std::string NP::descMeta() const 
{
    std::stringstream ss ; 
    ss << "NP::descMeta" 
       << std::endl 
       << meta 
       << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}

inline const char* NP::get_lpath() const 
{
    return lpath.c_str() ? lpath.c_str() : "-" ; 
}


/**
NP::set_meta
--------------

A preexisting keyed k:v pair is changed by this otherwise if there is no 
such pre-existing key a new k:v pair is added. 

**/
template<typename T> inline void NP::set_meta(const char* key, T value)  
{
    SetMeta(meta, key, value); 
}

template void     NP::set_meta<int>(const char*, int ); 
template void     NP::set_meta<unsigned>(const char*, unsigned ); 
template void     NP::set_meta<float>(const char*, float ); 
template void     NP::set_meta<double>(const char*, double ); 
template void     NP::set_meta<std::string>(const char*, std::string ); 

























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

/**
NP::Memcmp
-----------

* -1: array lengths differ
* 0:bytes of the two arrays match
* other value indicating the array bytes differ 

**/

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

        if(VERBOSE) std::cout << "NP::Concatenate " << std::setw(3) << i << " " << a->desc() << " nv " << nv << std::endl ; 
    }

    unsigned ni_total = 0 ; 
    for(unsigned i=0 ; i < aa.size() ; i++) ni_total += aa[i]->shape[0] ; 
    if(VERBOSE) std::cout << "NP::Concatenate ni_total " << ni_total << std::endl ; 

    std::vector<int> comb_shape ; 
    NPS::copy_shape( comb_shape, a0->shape );  
    comb_shape[0] = ni_total ; 

    NP* c = new NP(a0->dtype); 
    c->set_shape(comb_shape); 
    if(VERBOSE) std::cout << "NP::Concatenate c " << c->desc() << std::endl ; 

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
When all the n are equal the annotation and padding could be disabled by setting annotate=false.  

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
        assert( strcmp( a->dtype, dtype0 ) == 0 && "input arrays must all have same dtype" ); 

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

    if(VERBOSE) std::cout 
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

        // NB: a_bytes may be less than item_bytes 
        // effectively are padding to allow ragged arrays to be handled together

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
                if(VERBOSE) std::cout << "NP::Combine annotate " << i << " uif32.u  " << uif32.u  << std::endl ; 
                *(cc + (i+1)*item_bytes/ebyte0 - 1) = uif32.f ;   
                // (i+1)*item_bytes/ebyte0 is off the edge, then -1 to be the last value 
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
                if(VERBOSE) std::cout << "NP::Combine annotate " << i << " uif64.u  " << uif64.u  << std::endl ; 
                *(cc + (i+1)*item_bytes/ebyte0 - 1) = uif64.f ;   
            }  

            c->set_preserve_last_column_integer_annotation() ;  
            // make the annotation survive MakeNarrow  
            // (currently annotation is scrubbed by MakeWide but could be easily be implented)
        }
    }
    return c ; 
}


template<typename... Args> inline NP* NP::Combine(Args ... args)  // Combine_ellipsis
{
    std::vector<const NP*> aa = {args...};
    bool annotate = true ;  
    return Combine(aa, annotate); 
}



inline NP* NP::Load(const char* path)
{
    if(VERBOSE) std::cerr << "[ NP::Load " << path << std::endl ; 

    bool npy_ext = U::EndsWith(path, EXT) ; 
    NP* a = nullptr ; 
    if(npy_ext)
    {
        a  = NP::Load_(path);
    }  
    else
    {
        std::vector<std::string> nms ;
        U::DirList(nms, path, EXT);
        std::cout
            << "NP::Load"
            << " path " << path
            << " U::DirList contains nms.size " << nms.size()
            << " EXT " << EXT
            << std::endl
            ;
        a = NP::Concatenate(path, nms);
    }
    if(VERBOSE) std::cerr << "] NP::Load " << path << std::endl ; 
    return a ;
}

inline NP* NP::Load_(const char* path)
{
    if(!path) return nullptr ; 
    NP* a = new NP() ; 
    int rc = a->load(path) ; 
    return rc == 0 ? a  : NULL ; 
}

inline NP* NP::Load(const char* dir, const char* name)
{
    if(!dir) return nullptr ; 
    std::string path = form_path(dir, name); 
    return Load(path.c_str());
}

inline NP* NP::Load(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ; 
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



inline bool NP::Exists(const char* base, const char* rel,  const char* name) // static 
{
    std::string path = form_path(base, rel, name); 
    return Exists(path.c_str()); 
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
    if(VERBOSE) std::cerr << "[ NP::load " << path << std::endl ; 

    lpath = path ;  // loadpath 
    lfold = U::DirName(path); 

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
    load_names( path ); 

    if(VERBOSE) std::cerr << "] NP::load " << path << std::endl ; 
    return 0 ; 
}

inline int NP::load_string_( const char* path, const char* ext, std::string& str )
{
    std::string str_path = U::ChangeExt(path, ".npy", ext ); 
    std::ifstream fp(str_path.c_str(), std::ios::in);
    if(fp.fail()) return 1 ; 

    std::stringstream ss ;                       
    std::string line ; 
    while (std::getline(fp, line))
    {
        ss << line << std::endl ;   // getline swallows new lines  
    }
    str = ss.str(); 
    return 0 ; 
}

inline int NP::load_strings_( const char* path, const char* ext, std::vector<std::string>& vstr )
{
    std::string vstr_path = U::ChangeExt(path, ".npy", ext ); 
    std::ifstream fp(vstr_path.c_str(), std::ios::in);
    if(fp.fail()) return 1 ; 

    std::string line ; 
    while (std::getline(fp, line)) vstr.push_back(line);  // getline swallows new lines  
    return 0 ; 
}


inline int NP::load_meta(  const char* path ){  return load_string_( path, "_meta.txt",  meta  ) ; }
inline int NP::load_names( const char* path ){  return load_strings_( path, "_names.txt", names ) ; }


inline void NP::save_string_(const char* path, const char* ext, const std::string& str ) const 
{
    if(str.empty()) return ; 
    std::string str_path = U::ChangeExt(path, ".npy", ext ); 
    if(VERBOSE) std::cout << "NP::save_string_ str_path [" << str_path  << "]" << std::endl ; 
    std::ofstream fps(str_path.c_str(), std::ios::out);
    fps << str ;  
}

inline void NP::save_strings_(const char* path, const char* ext, const std::vector<std::string>& vstr ) const 
{
    if(vstr.size() == 0) return ; 
    std::string vstr_path = U::ChangeExt(path, ".npy", ext ); 
    if(VERBOSE) std::cout << "NP::save_strings_ vstr_path [" << vstr_path  << "]" << std::endl ; 

    char delim = '\n' ; 
    std::ofstream fps(vstr_path.c_str(), std::ios::out);
    for(unsigned i=0 ; i < vstr.size() ; i++) 
    {
        const std::string& str = vstr[i] ;  
        fps << str << delim ;  
    }
}


inline void NP::save_meta( const char* path) const { save_string_(path, "_meta.txt",  meta  );  }
inline void NP::save_names(const char* path) const { save_strings_(path, "_names.txt", names );  }


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
    if(VERBOSE) std::cout << "NP::save path [" << path  << "]" << std::endl ; 

    int rc = U::MakeDirsForFile(path); 
    assert( rc == 0 ); 

    std::string hdr = make_header(); 
    std::ofstream fpa(path, std::ios::out|std::ios::binary);
    fpa << hdr ; 
    fpa.write( bytes(), arr_bytes() );

    save_meta( path); 
    save_names(path); 
}

inline void NP::save(const char* dir, const char* reldir, const char* name) const 
{
    if(VERBOSE) std::cout << "NP::save dir [" << ( dir ? dir : "-" )  << "] reldir [" << ( reldir ? reldir : "-" )  << "] name [" << name << "]" << std::endl ; 
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

/**
NP::_dump
-----------


**/
template <typename T> inline void NP::_dump(int i0_, int i1_, int j0_, int j1_ ) const 
{
    int ni = NPS::ni_(shape) ;  // ni_ nj_ nk_ returns shape dimension size or 1 if no such dimension
    int nj = NPS::nj_(shape) ;
    int nk = NPS::nk_(shape) ;

    int i0 = i0_ == -1 ? 0                : i0_ ;  
    int i1 = i1_ == -1 ? std::min(ni, 10) : i1_ ;  

    int j0 = j0_ == -1 ? 0                : j0_ ;  
    int j1 = j1_ == -1 ? std::min(nj, 10) : j1_ ;  


    std::cout 
       << desc() 
       << std::endl 
       << " array dimensions " 
       << " ni " << ni 
       << " nj " << nj 
       << " nk " << nk
       << " item range  "
       << " i0 " << i0 
       << " i1 " << i1 
       << " j0 " << j0 
       << " j1 " << j1 
       << std::endl 
       ;  

    const T* vv = cvalues<T>(); 

    for(int i=i0 ; i < i1 ; i++){
        std::cout << "[" << std::setw(4) << i  << "] " ;
        for(int j=j0 ; j < j1 ; j++){
            for(int k=0 ; k < nk ; k++)
            {
                int index = i*nj*nk + j*nk + k ; 
                T v = *(vv + index) ; 
                if(k%4 == 0 ) std::cout << " : " ;  
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


template <typename T> void NP::read(const T* src) 
{
    T* v = values<T>(); 

    NPS sh(shape); 
    for(int i=0 ; i < sh.ni_() ; i++ ) 
    for(int j=0 ; j < sh.nj_() ; j++ )
    for(int k=0 ; k < sh.nk_() ; k++ )
    for(int l=0 ; l < sh.nl_() ; l++ )
    for(int m=0 ; m < sh.nm_() ; m++ )
    for(int o=0 ; o < sh.no_() ; o++ )
    {  
        int index = sh.idx(i,j,k,l,m,o); 
        *(v + index) = *(src + index ) ; 
    }   
}

template <typename T> void NP::read2(const T* src) 
{
    bool consistent = sizeof(T) == ebyte ; 
    if(!consistent) std::cout << "NP::read2 FAIL not consistent sizeof(T): " << sizeof(T) << " and ebyte: " << ebyte << std::endl ;  
    assert( consistent ); 
    memcpy( bytes(), src, arr_bytes() );    
}

template <typename T> 
inline void NP::write(T* dst) const 
{
    assert( sizeof(T) == ebyte ); 
    memcpy( dst, bytes(), arr_bytes() );    
}



template <typename T> 
inline NP* NP::Linspace( T x0, T x1, unsigned nx, int npayload ) 
{
    assert( x1 > x0 ); 
    assert( nx > 0 ) ; 
    NP* a = NP::Make<T>(nx, npayload );  // npayload default is -1

    if( nx == 1 )
    {
        a->set<T>(x0, 0 ); 
    }
    else
    {
        for(unsigned i=0 ; i < nx ; i++) a->set<T>( x0 + (x1-x0)*T(i)/T(nx-1), i )  ; 
    }
    return a ; 
}

/**
NP::MakeDiv
-------------

When applied to a 1d array the contents are assummed to be domain edges 
that are divided by an integer multiple *mul*. For a src array of length ni 
the output array length is::

    (ni - 1)*mul + 1  

When applied to a 2d array the contents are assumed to be (ni,2) with 
(domain,value) pairs. The domain is divided as in the 1d case and values
are filled in via linear interpolation.

For example, 

* mul=1 -> ni
* mul=2 -> (ni-1)*2+1 = 2*ni-1 
* mul=3 -> (ni-1)*3+1 = 3*ni-2 

That is easier to understand in terms of the number of bins:

* mul=1   ni-1 -> 1*(ni-1)
* mul=2   ni-1 -> 2*(ni-1) 
* mul=3   ni-1 -> 3*(ni-1) 

Avoids repeating the top sub-edge of one bin that is the same as the first sub-edge 
of the next bin by skipping the last sub-edge unless it is from the last bin. 


         +-----------------+     2 values, 1 bin    (mul 1)

         +--------+--------+     3 values, 2 bins   (mul 2)

         +----+---+---+----+     5 values, 4 bins   (mul 4)

         +--+-+-+-+-+-+--+-+     9 values, 8 bins   (mul 8)  

**/


template <typename T> 
inline NP* NP::MakeDiv( const NP* src, unsigned mul  )
{
    assert( mul > 0 ); 
    unsigned ndim = src->shape.size(); 
    assert( ndim == 1 || ndim == 2 ); 

    unsigned src_ni = src->shape[0] ; 
    unsigned src_bins = src_ni - 1 ; 
    unsigned dst_bins = src_bins*mul ;   

    int dst_ni = dst_bins + 1 ; 
    int dst_nj = ndim == 2 ? src->shape[1] : -1 ; 

#ifdef DEBUG
    std::cout 
        << " mul " << std::setw(3) << mul
        << " src_ni " << std::setw(3) << src_ni
        << " src_bins " << std::setw(3) << src_bins
        << " dst_bins " << std::setw(3) << dst_bins
        << " dst_ni " << std::setw(3) << dst_ni
        << " dst_nj " << std::setw(3) << dst_nj
        << std::endl
        ; 
#endif

    NP* dst = NP::Make<T>( dst_ni, dst_nj ); 
    T* dst_v = dst->values<T>(); 

    for(unsigned i=0 ; i < src_ni - 1 ; i++)
    {
        bool first_i = i == 0 ; 
        const T s0 = src->get<T>(i,0) ; 
        const T s1 = src->get<T>(i+1,0) ; 

#ifdef DEBUG
        std::cout 
            << " i " << std::setw(3) << i 
            << " first_i " << std::setw(1) << first_i 
            << " s0 " << std::setw(10) << std::fixed << std::setprecision(4) << s0
            << " s1 " << std::setw(10) << std::fixed << std::setprecision(4) << s1
            << std::endl  
            ; 
#endif
        for(unsigned s=0 ; s < 1+mul ; s++) // s=0,1,2,... mul 
        {
            bool first_s = s == 0 ; 
            if( first_s && !first_i ) continue ;  // avoid repeating idx from bin to bin  

            const T frac = T(s)/T(mul) ;    //  frac(s=0)=0  frac(s=mul)=1   
            const T ss = s0 + (s1 - s0)*frac ;  
            unsigned idx = i*mul + s ; 

#ifdef DEBUG
            std::cout 
                << " s " << std::setw(3) << s 
                << " first_s " << std::setw(1) << first_s
                << " idx " << std::setw(3) << idx
                << " ss " << std::setw(10) << std::fixed << std::setprecision(4) << ss 
                << std::endl  
                ; 
#endif

            assert( idx < dst_ni ); 
    
            if( dst_nj == -1 )
            {
                dst_v[idx] = ss ; 
            }
            else if( dst_nj == 2 )
            {
                dst_v[2*idx+0] = ss ; 
                dst_v[2*idx+1] = src->interp<T>(ss) ; 
            }
        }
    }
    return dst ; 
}


template <typename T> 
inline NP*  NP::Make( const std::vector<T>& src ) // static
{
    NP* a = NP::Make<T>(src.size()); 
    a->read(src.data()); 
    return a ; 
}

template <typename T> 
inline NP*  NP::Make(T d0, T v0, T d1, T v1 ) // static
{
    std::vector<T> src = {d0, v1, d1, v1 } ; 
    return NP::Make<T>(src) ; 
}




template <typename T> NP* NP::FromString(const char* str, char delim)  // static 
{   
    std::vector<T> vec ; 
    std::stringstream ss(str);
    std::string s ; 
    while(getline(ss, s, delim)) vec.push_back(To<T>(s.c_str()));
    NP* a = NP::Make<T>(vec) ; 
    return a ; 
}











template <typename T> unsigned NP::NumSteps( T x0, T x1, T dx )
{
    assert( x1 > x0 ); 
    assert( dx > T(0.) ) ; 

    unsigned ns = 0 ; 
    for(T x=x0 ; x <= x1 ; x+=dx ) ns+=1 ; 
    return ns ; 
}


template <typename T> NP* NP::Make( int ni_, int nj_, int nk_, int nl_, int nm_, int no_ )
{
    std::string dtype = descr_<T>::dtype() ; 
    NP* a = new NP(dtype.c_str(), ni_,nj_,nk_,nl_,nm_, no_) ;    
    return a ; 
}


template<typename T> NP* NP::MakeFlat(int ni, int nj, int nk, int nl, int nm, int no )
{
    NP* a = NP::Make<T>(ni, nj, nk, nl, nm, no );  
    a->fillIndexFlat(); 
    return a ; 
}






/**
NP::Make "Make_ellipsis"
--------------------------

This "Make_ellipsis" method combines allocation of the array and populating it 
from the src data. This is intended to facilitate creating arrays from vectors
of struct, by using simple template types  (int, float, double etc.. )  
together with array item shapes appropriate to the elements of the struct. 
For example::

   struct demo { int x,y,z,w ; } ; 
   std::vector<demo> dd ; 
   dd.push_back( {1,2,3,4} ); 

   NP* a = NP::Make<int>( (int*)dd.data() , int(dd.size()) , 4 ); 

The product of the shape integers MUST correspond to the number of 
values provided from the src data. 
When the first int shape dimension is zero a nullptr is returned.

**/

template<typename T, typename... Args> NP* NP::Make(const T* src, Args ... args )   // Make_ellipsis
{
    std::string dtype = descr_<T>::dtype() ; 
    std::vector<int> shape = {args...};
    if(shape.size() > 0 && shape[0] == 0) return nullptr ; 
    NP* a = new NP(dtype.c_str(), shape ); 
    a->read2(src);  
    return a ; 
}




template <typename T> void NP::Write(const char* dir, const char* reldir, const char* name, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_, int no_ ) // static
{
    std::string path = form_path(dir, reldir, name); 
    Write( path.c_str(), data, ni_, nj_, nk_, nl_, nm_, no_ ); 
}

template <typename T> void NP::Write(const char* dir, const char* name, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_, int no_ ) // static
{
    std::string path = form_path(dir, name); 
    Write( path.c_str(), data, ni_, nj_, nk_, nl_, nm_, no_ ); 
}

template <typename T> void NP::Write(const char* path, const T* data, int ni_, int nj_, int nk_, int nl_, int nm_, int no_ ) // static
{
    std::string dtype = descr_<T>::dtype() ; 
    if(VERBOSE) std::cout 
        << "NP::Write"
        << " dtype " << dtype
        << " ni  " << std::setw(7) << ni_
        << " nj  " << nj_
        << " nk  " << nk_
        << " nl  " << nl_
        << " nm  " << nm_
        << " no  " << no_
        << " path " << path
        << std::endl 
        ;   

    if(ni_ == 0) return ; 
    NP a(dtype.c_str(), ni_,nj_,nk_,nl_,nm_,no_) ;    
    a.read(data); 
    a.save(path); 
}




template void NP::Write<float>(   const char*, const char*, const float*,        int, int, int, int, int, int ); 
template void NP::Write<double>(  const char*, const char*, const double*,       int, int, int, int, int, int ); 
template void NP::Write<int>(     const char*, const char*, const int*,          int, int, int, int, int, int ); 
template void NP::Write<unsigned>(const char*, const char*, const unsigned*,     int, int, int, int, int, int ); 


template<typename T> void NP::Write(const char* dir, const char* name, const std::vector<T>& values )
{
    if(values.size() > 0) NP::Write(dir, name, values.data(), values.size() ); 
}

template void NP::Write<float>(   const char*, const char*, const std::vector<float>& ); 
template void NP::Write<double>(  const char*, const char*, const std::vector<double>&  ); 
template void NP::Write<int>(     const char*, const char*, const std::vector<int>& ); 
template void NP::Write<unsigned>(const char*, const char*, const std::vector<unsigned>& ); 

/*
HMM: Read not so easy to encapsulate as normally using vectors of many different types of compound types 

template<typename T> void NP::Read(const char* dir, const char* name, T* dst )
{
    NP* a = NP::Load(dir, name); 
    values.resize(a->shape[0])
    memcpy( (T*)dst,    a->cvalues<T>() ,  a->arr_bytes() );
}
*/






inline void NP::WriteNames(const char* dir, const char* name, const std::vector<std::string>& names, unsigned num_names_, bool append )
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    WriteNames(path.c_str(), names, num_names_, append  ); 
}


inline void NP::WriteNames(const char* dir, const char* reldir, const char* name, const std::vector<std::string>& names, unsigned num_names_, bool append )
{
    std::string path = form_path(dir, reldir, name); 
    WriteNames(path.c_str(), names, num_names_, append ); 
}

/**
NP::WriteNames
----------------

https://stackoverflow.com/questions/12929378/what-is-the-difference-between-iosapp-and-iosate

app : 'append' 
    all output will be added (appended) to the end of the file. 
    In other words you cannot write anywhere else in the file but at the end.

ate : 'at end' 
    sets the stream position at the end of the file when you open it, 
    but you are free to move it around (seek) and write wherever it pleases you.

**/

inline void NP::WriteNames(const char* path, const std::vector<std::string>& names, unsigned num_names_, bool append )
{
    if(names.size() == 0) return ; 
    int rc = U::MakeDirsForFile(path); 
    assert( rc == 0 ); 

    unsigned num_names = num_names_ == 0 ? names.size() : num_names_ ; 
    assert( num_names <= names.size() ); 

    std::ios_base::openmode mode = std::ios::out|std::ios::binary ; 
    if(append) mode |= std::ios::app ;

    std::ofstream stream(path, mode );
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





template<typename T>
inline std::string NP::DescKV(const std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extras)
{
    std::stringstream ss ; 
    assert( keys.size() == vals.size() ); 
    if(extras) assert( extras->size() == keys.size() );
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
         ss 
            << std::setw(20) << keys[i] 
            << " : "
            << std::scientific << std::setw(10) << std::setprecision(5) << vals[i]
            << " : "
            << ( extras ? (*extras)[i] : "" )
            << std::endl   
            ;   
    }
    std::string s = ss.str(); 
    return s ; 
}


template<typename T>
inline void NP::ReadKV(const char* dir, const char* name, std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extras ) 
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    ReadKV(path.c_str(), keys, vals, extras); 
}

template<typename T>
inline void NP::ReadKV(const char* path, std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extras ) 
{
    std::ifstream ifs(path);
    std::string line;
    while(std::getline(ifs, line)) 
    {
        std::string key ; 
        T val ; 
        std::string extra ; 

        std::istringstream iss(line);
        iss >> key >> val >> extra ; 

        if(VERBOSE) std::cout 
            << "NP::ReadKV" 
            << " key[" <<  key << "]" 
            << " val[" <<  val << "]" 
            << " extra[" <<  extra << "]" 
            << std::endl ;  

        keys.push_back(key); 
        vals.push_back(val); 
        if(extras) extras->push_back(extra); 
    }
}

template<typename T>
inline T NP::ReadKV_Value(const char* dir, const char* name, const char* key )
{
    std::stringstream ss ; 
    ss << dir << "/" << name ; 
    std::string path = ss.str() ; 
    return NP::ReadKV_Value<T>(path.c_str(), key ); 
}

template<typename T>
inline T NP::ReadKV_Value(const char* spec_or_path, const char* key)
{
    const char* path = Resolve(spec_or_path); 

    std::vector<std::string> keys ; 
    std::vector<T> vals ; 
    std::vector<std::string> extras ; 

    ReadKV<T>(path, keys, vals, &extras ); 

    std::vector<std::string>::iterator it = std::find(keys.begin(), keys.end(), key ) ; 

    if(it == keys.end())
    {
        std::cout 
            << "NP::ReadKV_Value"
            << " FATAL "
            << " failed to find key " << key 
            << std::endl 
            ;
        std::cout << NP::DescKV<T>(keys, vals, &extras ) << std::endl ; 
        assert(0); 
    }

    unsigned idx = std::distance( keys.begin(), it ); 
    return vals[idx] ;  
}



template <typename T> 
inline NP* NP::LoadFromTxtFile(const char* base, const char* relp )  // static 
{   
    std::stringstream ss ;  
    ss << base << "/" << relp ; 
    std::string path = ss.str(); 
    NP* a = LoadFromTxtFile<T>( path.c_str()); 
    a->lpath = path ; 
    return a ; 
}


/**
NP::LoadFromTxtFile
----------------------

1. resolves spec_or_path into path
2. reads txt from the file into str
3. creates array with NP::ArrayFromString  

**/

template <typename T> 
inline NP* NP::LoadFromTxtFile(const char* spec_or_path )  // static 
{   
    const char* path = Resolve(spec_or_path ) ; 
    if(!Exists(path))
    {
        std::cerr 
            << "NP::ArrayFromTxtFile"
            << " FATAL path does not EXIST "
            << " spec_or_path [" << spec_or_path << "]"
            << " path [" << path << "]"
            << std::endl 
            ;
        assert(0); 
    }

    const char* str = ReadString2(path); 
    NP* a = LoadFromString<T>(str, path); 
    a->lpath = path ; 
    return a ; 
}

/**
NP::FindUnit
--------------

Each unit string is looked for within the line, 
the last in the units list that matches is returned. 

**/

inline char* NP::FindUnit(const char* line, const std::vector<std::string>& units  ) // static
{
    char* upos = nullptr ; 
    for(unsigned i=0 ; i < units.size() ; i++)
    {
        const char* u = units[i].c_str(); 
        upos = (char*)strstr(line, u) ; 
    } 
    return upos ; 
}

inline void NP::GetUnits(std::vector<std::string>& units ) // static
{
    std::stringstream uss(UNITS) ;
    std::string unit ; 
    while(std::getline(uss,unit,' ')) units.push_back(unit) ;   
}
inline bool NP::IsListed(const std::vector<std::string>& ls, const char* str) // static
{
    return std::find(ls.begin(), ls.end(), str ) != ls.end() ; 
} 
inline std::string NP::StringConcat(const std::vector<std::string>& ls, char delim ) // static
{
    unsigned num = ls.size() ; 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++ ) 
    {
        ss << ls[i] ; 
        if( i < num - 1) ss << delim ; 
    }
    std::string cls = ss.str() ; 
    return cls ; 
}

template <typename T> 
inline NP* NP::ZEROProp()  // static 
{ 
    return NP::LoadFromString<T>(R"(
    1.55     *eV    0.0
    15.5     *eV    0.0
)" ); 

}


/**
NP::LoadFromString
----------------------

String format example::

   ScintillationYield   9846/MeV
   BirksConstant1  12.05e-3*g/cm2/MeV

Each line is cleaned to correct the poor file format 
regaining whitespace between fields:

1. '/' prior to recognized unit strings are changed to ' ' 
2. all '*' are changed to ' ' 

After cleanup, the number of fields on each line must be consistent
for all lines of the string. Also the number of fields that 
can be converted to type T must be consistent for all lines. 

So for the above example an array of shape (2,1) would be created 
with a names vector containing the non-unit strings, which 
allowed named access to values with NP::get_named_value 

Another example, the below input txt with type "float" or "double"::

    1.55     *eV    2.72832
    2.69531  *eV    2.7101
    2.7552   *eV    2.5918
    3.17908  *eV    1.9797
    15.5     *eV    1.9797

would yield an array of shape (5,2) with metadata key "unit" of "eV" 

**/


template <typename T> 
inline NP* NP::LoadFromString(const char* str, const char* path)  // static 
{ 
    // path is optional for debug messages
    //std::cout << "NP::LoadFromString " << ( path ? path : "-" ) << std::endl ; 

    std::vector<std::string> recognized_units ; 
    GetUnits(recognized_units); 

    unsigned UNSET = ~0u ; 
    unsigned num_field = UNSET ; 
    unsigned num_column = UNSET ; 

    std::vector<std::string> units ; 
    std::vector<std::string> other ; 
    std::vector<T> value ; 

    std::string line ; 
    std::stringstream fss(str) ;
    while(std::getline(fss, line)) 
    {
        char* l = (char*)line.c_str() ;

        if(strlen(l) == 0) continue ; 
        if(strlen(l) > 0 && l[0] == '#') continue ; 



        // if a unit string is found which is preceeded by '/' remove that 
        // to regain whitespace between fields 
        //
        char* upos = FindUnit(l, recognized_units) ; 
        if(upos && (upos - l) > 0)
        {
            if(*(upos-1) == '/') *(upos-1) = ' ' ;   
        } 

        ReplaceCharInsitu( l, '*', ' ', false ); 


        std::vector<std::string> fields ; 
        std::string field ; 
        std::istringstream iss(line);
        while( iss >> field ) 
        {
            const char* f = field.c_str(); 
            if(IsListed(recognized_units, f))
            {
                if(!IsListed(units, f)) units.push_back(f); 
            }
            else
            {
                fields.push_back(field) ; 
            }
        }

        if(fields.size() == 0) continue ; 

        if( num_field == UNSET ) 
        {
            num_field = fields.size() ; 
        }
        else if( fields.size() != num_field )
        {
            std::cerr 
                << "NP::ArrayFromString" 
                << " WARNING : INCONSISTENT NUMBER OF FIELDS " << std::endl 
                << " [" << line << "]" << std::endl 
                << " fields.size : " << fields.size() 
                << " num_field : " << num_field 
                << " path " << ( path ? path : "-" )
                << std::endl
                ;
            assert(0); 
        }
        assert( num_field != UNSET ); 

        //std::cout << "[" << line << "] num_field " << num_field << std::endl; 

        unsigned line_column = 0u ;  
        for(unsigned i=0 ; i < num_field ; i++)
        {
            const char* str = fields[i].c_str(); 
            if(ConvertsTo<T>(str)) 
            {   
                value.push_back(To<T>(str)) ; 
                line_column += 1 ;  
            }
            else
            {
                if(!IsListed(other, str)) other.push_back(str); 
            }
        }
   
        if( num_column == UNSET )
        {
            num_column = line_column ; 
        }
        else if( line_column != num_column )
        {
            std::cerr
                << "NP::ArrayFromString"
                << " FATAL : INCONSISTENT NUMBER OF VALUES " << std::endl  
                << " [" << line << "]" << std::endl 
                << " fields.size : " << fields.size() 
                << " num_field : " << num_field 
                << " num_column : " << num_column
                << " line_column : " << line_column
                << " path " << ( path ? path : "-" )
                << std::endl
                ;
            assert(0); 
        }
    }

    unsigned num_value = value.size() ; 
    assert( num_value % num_column == 0 ); 

    unsigned num_row = num_value/num_column ; 
    assert( num_row*num_column == num_value ); 

    NP* a = NP::Make<T>( num_row, num_column ); 
    a->read2( value.data() ); 

 
    if(units.size() > 0)
    {
        //for(unsigned i=0 ; i < units.size() ; i++ ) std::cout << "units[" << units[i] << "]" << std::endl  ; 
        std::string u_units = StringConcat(units, ' '); 
        a->set_meta<std::string>("units", u_units ); 
    }

    if(other.size() > 0)
    {
        //for(unsigned i=0 ; i < other.size() ; i++ ) std::cout << "other[" << other[i] << "]" << std::endl  ; 
        std::string u_other = StringConcat(other, ' '); 
        a->set_meta<std::string>("other", u_other ); 

        if( num_column == 1 && other.size() == num_row ) a->set_names(other) ; 
    }
    return a ; 
}




inline unsigned NP::CountChar(const char* str, char q )
{
    unsigned count = 0 ; 
    char* c = (char*)str ; 
    while(*c)
    {
        if(*c == '.') count += 1 ; 
        c++ ; 
    }  
    return count ; 
}

inline void NP::ReplaceCharInsitu(char* str, char q, char n, bool first )
{
    unsigned count = 0 ;  
    char* c = str ; 
    while(*c)
    {
        if(*c == q) 
        {
           if((first && count == 0) || first == false ) *c = n ; 
           count += 1 ; 
        }
        c++ ; 
    }     
}
inline const char* NP::ReplaceChar(const char* str, char q, char n, bool first  )
{
    char* s = strdup(str); 
    ReplaceCharInsitu(s, q, n, first ); 
    return s ; 
}

inline const char* NP::Resolve( const char* spec) 
{
    return CountChar(spec, '.') > 1 ? ResolveProp(spec) : spec ; 
}

inline const char* NP::ResolveProp(const char* spec)
{
    assert(CountChar(spec, '.') > 1);

    char* s = strdup(spec) ; 
    while(*s && *s == ' ') s++ ;  // skip any leading whitespace 
    char* c = s ; 
    while(*c)    // terminate when hit end of spec or trailing whitespace
    {
        if(*c == '.') *c = '/' ; 
        c++ ; 
        if(*c == ' ') *c = '\0' ;  // terminate at first trailing space
    }
    const char* base = getenv("NP_PROP_BASE") ; 
    std::stringstream ss ; 
    ss << ( base ? base : "/tmp" ) << "/" << s  ; 

    std::string path = ss.str(); 
    return strdup(path.c_str()) ; 
}


inline void NP::WriteString( const char* dir, const char* reldir, const char* name, const char* str )  // static 
{
    std::string path = form_path(dir, reldir, name); 
    WriteString(path.c_str(), str); 
}

inline void NP::WriteString( const char* dir, const char* name, const char* str )  // static 
{
    std::string path = form_path(dir, name); 
    WriteString(path.c_str(), str); 
}

inline void NP::WriteString( const char* path, const char* str )  // static 
{
    if(str == nullptr) return ; 
    std::ofstream fp(path, std::ios::out);
    fp << str ; 
    fp.close(); 
}


inline const char* NP::ReadString( const char* dir, const char* reldir, const char* name) // static
{
    std::string path = form_path(dir, reldir, name); 
    return ReadString(path.c_str()); 
}

inline const char* NP::ReadString( const char* dir, const char* name) // static
{
    std::string path = form_path(dir, name); 
    return ReadString(path.c_str()); 
}

inline const char* NP::ReadString( const char* path )  // static
{
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

inline const char* NP::ReadString2(const char* path)  // static
{
    std::ifstream ifs(path);
    std::stringstream ss ;
    ss << ifs.rdbuf();
    std::string str = ss.str(); 
    return str.empty() ? nullptr : strdup(str.c_str()) ; 
}



inline std::string NP::descValues() const 
{
    assert( shape.size() == 1 ); 
    unsigned num_val = shape[0] ; 
    assert( names.size() == num_val );
    assert( ebyte == 8 );  
    std::stringstream ss ; 
    ss << "NP::descValues num_val " << num_val  << std::endl ; 
    const double* vv = cvalues<double>() ; 
    for(unsigned i=0 ; i < num_val ; i++)
    {
        const char* k = names[i].c_str(); 
        ss
            << std::setw(3) << i 
            << " v " << std::setw(10) << std::fixed << std::setprecision(4) << vv[i] 
            << " k " << std::setw(60) << std::left << k << std::right 
            <<  std::endl 
            ;
    }
    std::string s = ss.str(); 
    return s ; 
}





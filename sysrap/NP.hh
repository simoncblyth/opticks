#ifndef NP_HH
#define NP_HH

/**
NP : Header-only Array Creation and persisting as NumPy .npy files
====================================================================

* TODO: relocate higher level NP.hh functionality up to NPX.h
* TODO: relocate lower level NP.hh functionality down to NPU.hh


Dependencies of the NP family of headers::

    NPFold.h : NPX.h

    NPX.h : NP.hh

    NP.hh : NPU.hh

    NPU.hh : system headers only



NPU.hh
    underpinnings of NP.hh
NP.hh
    core of save/load arrays into .npy NumPy format files
NPX.h
    extras such as static converters
NPFold.h
    managing and persisting collections of arrays and other NPFold

Primary source is https://github.com/simoncblyth/np/
but the headers are also copied into opticks/sysrap.

**/

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>
#include <csignal>
#include <fstream>
#include <cstdint>
#include <limits>
#include <random>
#include <map>
#include <functional>
#include <locale>

#include "NPU.hh"


struct NP
{
    typedef std::int64_t INT ;
    static constexpr const INT TEN = 10 ;

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


    // SPECIALIZED MEMBER FUNCTIONS

    template<typename T> const T*  cvalues() const  ;
    template<typename T> T*       values() ;

    template<typename T> void fill(T value);
    template<typename T> void _fillIndexFlat(T offset=0);

    // BLOCK OF TEMPLATE SPECIALIZATIONS cvalues, values, _fillIndexFlat : IN IMPL BELOW AT THIS POINT


    // STATIC CREATION METHODS

    template<typename T> static NP* MakeFromValues( const T* vals, INT num_vals );
    template<typename T> static INT ALength(  T x0, T x1, T st );
    template<typename T> static NP* ARange(   T x0, T x1, T st );
    template<typename T> static NP* Linspace( T x0, T x1, unsigned nx, INT npayload=-1 );
    template<typename T> static NP* DeltaColumn(const NP* a, INT jcol=0 ) ;

    template<typename T> static NP* MinusCosThetaLinearAngle(INT nx=181); // from -1. to 1.
    template<typename T> static NP* ThetaRadians(INT nx=181, T theta_max_pi=1. ); // from 0. to theta_max_pi*PI
                         static NP* SqrtOneMinusSquare( const NP* a );
                         static NP* Cos( const NP* src );
                         static NP* MakeWithCosineDomain( const NP* src, bool reverse );
                         static NP* Incremented( const NP* a, INT offset  );

    template<typename T> static NP* MakeDiv( const NP* src, unsigned mul  );

    template<typename T> static NP* Make( INT ni_=-1, INT nj_=-1, INT nk_=-1, INT nl_=-1, INT nm_=-1, INT no_=-1 );
    template<typename T, typename ... Args> static NP*  Make_( Args ... shape ) ;  // Make_shape
    template<typename T> static NP* MakeFlat(INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 );


    //  MEMBER FUNCTIONS

    char*       bytes();
    const char* bytes() const ;

    INT hdr_bytes() const ;
    INT num_items() const ;       // shape[0]
    INT num_values() const ;      // all values, product of shape[0]*shape[1]*...
    INT num_itemvalues() const ;  // values after first dimension
    INT arr_bytes() const ;       // formerly num_bytes
    INT item_bytes() const ;      // *item* comprises all dimensions beyond the first
    INT meta_bytes() const ;

    template<typename T> bool is_itemtype() const ;  // size of item matches size of type

    void clear() ;

    void        update_headers();
    std::string make_header() const ;
    std::string make_prefix() const ;
    std::string make_jsonhdr() const ;

    bool        decode_header() ; // sets shape based on arr header
    bool        decode_prefix() ; // also resizes buffers ready for reading in
    unsigned    prefix_size(unsigned index) const ;



    // CTOR
    NP(const char* dtype_, const std::vector<INT>& shape_ );
    NP(const char* dtype_="<f4", INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 );

    void init();
    void set_shape( INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1);
    void set_shape( const std::vector<INT>& src_shape );
    // CAUTION: DO NOT USE *set_shape* TO CHANGE SHAPE (as it calls *init*) INSTEAD USE *change_shape*
    bool has_shape(INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 ) const ;
    void change_shape(INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 ) ;   // one dimension entry left at -1 can be auto-set
    void change_shape_to_3D() ;
    void reshape( const std::vector<INT>& new_shape ); // product of shape before and after must be the same

    template<int P> void size_2D( INT& width, INT& height ) const ;


    void set_dtype(const char* dtype_); // *set_dtype* may change shape and size of array while retaining the same underlying bytes


    INT index(  INT i,  INT j=0,  INT k=0,  INT l=0, INT m=0, INT o=0) const ;
    INT index0( INT i,  INT j=-1, INT k=-1,  INT l=-1, INT m=-1, INT o=-1) const ;

    INT dimprod(unsigned q) const ;    // product of dimensions starting from dimension q

    template<typename... Args>
    INT index_(Args ... idxx ) const ;

    template<typename... Args>
    INT stride_(Args ... idxx ) const ;

    template<typename... Args>
    INT offset_(Args ... idxx ) const ;


    template<typename T>
    static std::string ArrayString(const std::vector<T>& vec, unsigned modulo=10 );

    template<typename T, typename... Args>
    std::string sliceArrayString(Args ... idxx ) const ;

    // use -1 to mark the last dimension to select upon
    // eg to select first item use (0, -1)
    //
    // Like think NumPy indexing

    template<typename T, typename... Args>
    void slice(std::vector<T>& out, Args ... idxx ) const ;  // slice_ellipsis


    template<typename T>
    void slice_(std::vector<T>& out, const std::vector<INT>& idxx ) const ;

    template<typename T>
    static std::string DescSlice(const std::vector<T>& out, unsigned edge );

    template<typename T>
    static std::string DescSliceBrief(const std::vector<T>& out);


    static std::string DescIdx(const std::vector<INT>& idxx );


    INT pickdim__(    const std::vector<INT>& idxx) const ;

    INT index__( const std::vector<INT>& idxx) const ;
    INT stride__(const std::vector<INT>& idxx) const ;
    INT offset__(const std::vector<INT>& idxx) const ;



    INT       itemsize_(INT i=-1, INT j=-1, INT k=-1, INT l=-1, INT m=-1, INT o=-1) const ;
    void      itembytes_(const char** start,  INT& num_bytes, INT i=-1, INT j=-1, INT k=-1, INT l=-1, INT m=-1, INT o=-1 ) const  ;

    template<typename T> T           get( INT i,  INT j=0,  INT k=0,  INT l=0, INT m=0, INT o=0) const ;
    template<typename T> void        set( T val, INT i,  INT j=0,  INT k=0,  INT l=0, INT m=0, INT o=0 ) ;

    template<typename T> bool is_allzero() const ;
    bool is_empty() const ;


    std::string descValues() const ;
    std::string descSize() const ;



    template<typename T>
    std::string descTable(int wid=7) const ;

    template<typename T>
    T findMinimumTimestamp() const ;

    template<typename T>
    std::string descTable_(
       int wid=7,
       const std::vector<std::string>* column_labels=nullptr,
       const std::vector<std::string>* row_labels=nullptr
       ) const ;

    static NP* MakeLike(  const NP* src);
    static void CopyMeta( NP* b, const NP* a );

    static constexpr const char* Preserve_Last_Column_Integer_Annotation = "Preserve_Last_Column_Integer_Annotation" ;
    void set_preserve_last_column_integer_annotation();
    bool is_preserve_last_column_integer_annotation() const ;
    static float PreserveNarrowedDoubleInteger( double f );

    // STATIC CONVERSION METHODS

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
    static NP* MakeSelectCopy_( const NP* src, const std::vector<INT>* items );
    static NP* MakeSelectCopy_( const NP* src, const INT* items, INT num_items );

    static NP* MakeSelection( const NP* src, const NP* sel );  // sel expected to contain integer indices selecting items in src

    static int ParseSliceString(std::vector<INT>& idxx, const char* _sli );
    template<typename T> static NP* MakeSliceSelection( const NP* src, const char* sel );
    template<typename T> NP* makeWhereSelection( const char* _sel ) const ;


    static NP* MakeItemCopy(  const NP* src, INT i,INT j=-1,INT k=-1,INT l=-1,INT m=-1, INT o=-1 );
    void  item_shape(std::vector<INT>& sub, INT i, INT j=-1, INT k=-1, INT l=-1, INT m=-1, INT o=-1 ) const ;
    NP*   spawn_item(  INT i, INT j=-1, INT k=-1, INT l=-1, INT m=-1, INT o=-1  ) const ;

    template<typename T> static NP* MakeCDF(  const NP* src );
    template<typename T> static NP* MakeICDF(  const NP* src, unsigned nu, unsigned hd_factor, bool dump );
    template<typename T> static NP* MakeProperty(const NP* a, unsigned hd_factor );
    template<typename T> static NP* MakeLookupSample(const NP* icdf_prop, unsigned ni, unsigned seed=0u, unsigned hd_factor=0u );
    template<typename T> static NP* MakeUniform( unsigned ni, unsigned seed=0u );

    NP* copy() const ;

    template<typename S>                               INT         count_if( std::function<bool(const S*)>) const ;
    template<typename T>                               NP*   simple_copy_if( std::function<bool(const T*)>) const ;  // atomic types only
    template<typename T, typename S>                   NP*          copy_if( std::function<bool(const S*)>) const ;
    template<typename T, typename S, typename... Args> NP* flexible_copy_if( std::function<bool(const S*)>, Args ... itemshape ) const ;


    // load array asis
    static NP* LoadIfExists(const char* path);
    static NP* Load(const char* path);

    template<typename T> static NP* LoadSlice( const char* path, const char* _sel );

    static bool ExistsArrayFolder(const char* path );

    static NP* Load_(const char* path);
    static NP* Load(const char* dir, const char* name);
    static NP* Load(const char* dir, const char* reldir, const char* name);

    // load float OR double array and if float(4 bytes per element) widens it to double(8 bytes per element)
    static NP* LoadWide(const char* dir, const char* reldir, const char* name);
    static NP* LoadWide(const char* dir, const char* name);
    static NP* LoadWide(const char* path);

    // load float OR double array and if double(8 bytes per element) narrows it to float(4 bytes per element)
    static NP* LoadNarrow(const char* dir, const char* reldir, const char* name);
    static NP* LoadNarrow(const char* dir, const char* name);
    static NP* LoadNarrow(const char* path);

    template<typename T> INT find_value_index(T value, T epsilon) const ;
    template<typename T> T   ifind2D(T value, INT jcol, INT jret ) const ;


    bool is_pshaped() const ;
    template<typename T> bool is_pconst() const ;
    template<typename T> bool is_pconst_dumb() const ;  // constant prop with more that 2 items
    template<typename T> T    pconst(T fallback=-1) const ;

    template<typename T>
    static NP* MakePCopyNotDumb(const NP* a);

    template<typename T>
    static NP* MakePConst( T dl, T dr, T vc );


    template<typename T> T    plhs(unsigned column ) const ;
    template<typename T> T    prhs(unsigned column ) const ;
    template<typename T> INT  pfindbin(const T value, unsigned column, bool& in_range ) const ;
    template<typename T> void get_edges(T& lo, T& hi, unsigned column, INT ibin) const ;


    template<typename T> T    psum(unsigned column ) const ;
    template<typename T> void pscale(T scale, unsigned column);
    template<typename T> void pscale_add(T scale, T add, unsigned column);
    template<typename T> void pdump(const char* msg="NP::pdump", T d_scale=1., T v_scale=1.) const ;

    template<typename T> void minmax(T& mn, T&mx, unsigned j=1, INT item=-1 ) const ;
    template<int N, typename T> void minmax2D_reshaped(T* mn, T* mx, INT item_stride=1, INT item_offset=0) ; // not-const as temporarily changes shape
    template<typename T>        void minmax2D(T* mn, T* mx, INT item_stride=1, INT item_offset=0 ) const ;

    template<typename T> void linear_crossings( T value, std::vector<T>& crossings ) const ;
    template<typename T> NP*  trapz() const ;                      // composite trapezoidal integration, requires pshaped

    template<typename T> void psplit(std::vector<T>& domain, std::vector<T>& values) const ;
    template<typename T> T    pdomain(const T value, INT item=-1, bool dump=false  ) const ;
    template<typename T> T    interp(T x, INT item=-1) const ;                  // requires pshaped
    template<typename T> T    interp2D(T x, T y, INT item=-1) const ;


    template<typename T> T    interpHD(T u, unsigned hd_factor, INT item=-1 ) const ;

    template<typename T> T    interp(INT iprop, T x) const ;           // deprecated signature for combined_interp
    template<typename T> T    combined_interp_3(INT i,               T x) const ;  // requires NP::Combine of pshaped arrays
    template<typename T> T    combined_interp_5(INT i, INT j, INT k, T x) const ;  // requires NP::Combine of pshapes arrays

    template<typename T> T    _combined_interp(const T* vv, INT niv, T x) const  ;

    template<typename T> static T FractionalRange( T x, T x0, T x1 );


    template<typename T> NP*  cumsum(INT axis=0) const ;
    template<typename T> void divide_by_last() ;
    void fillIndexFlat();
    void dump(INT i0=-1, INT i1=-1, INT j0=-1, INT j1=-1) const ;

    static std::string Brief(const NP* a);
    std::string sstr() const ;
    std::string desc() const ;
    std::string brief() const ;

    template<typename T> std::string repr() const ;


    void set_meta( const std::vector<std::string>& lines, char delim='\n' );
    void get_meta( std::vector<std::string>& lines,       char delim='\n' ) const ;

    void set_names( const std::vector<std::string>& lines ) ;
    void get_names( std::vector<std::string>& lines ) const ;

    INT  get_name_index( const char* qname ) const ;
    INT  get_name_index( const char* qname, unsigned& count ) const ;
    static INT NameIndex( const char* qname, unsigned& count, const std::vector<std::string>& names );

    bool is_named_shape() const ;
    template<typename T> T  get_named_value( const char* qname, T fallback ) const ;

    bool has_meta() const ;
    static std::string               get_meta_string_(const char* metadata, const char* key);
    static std::string               get_meta_string( const std::string& meta, const char* key) ;

    typedef std::vector<std::string> VS ;
    typedef std::vector<int64_t> VT ;


    NP* makeMetaKVProfileArray(const char* ptn=nullptr) const ;
    static void GetMetaKV_( const char* metadata    , VS* keys, VS* vals, bool only_with_profile, const char* ptn=nullptr );
    static void GetMetaKV(  const std::string& meta , VS* keys, VS* vals, bool only_with_profile, const char* ptn=nullptr );

    template<typename T> static T    GetMeta( const std::string& mt, const char* key, T fallback );

    template<typename T> static T    get_meta_(const char* metadata, const char* key, T fallback=0) ;  // for T=std::string must set fallback to ""
    template<typename T> T    get_meta(const char* key, T fallback=0) const ;  // for T=std::string must set fallback to ""

    template<typename T> static void SetMeta(       std::string& mt, const char* key, T value );
    template<typename T> void set_meta(const char* key, T value ) ;

    template<typename T> void        set_meta_kv(                  const std::vector<std::pair<std::string, T>>& kvs );
    template<typename T> static void        SetMetaKV( std::string& meta, const std::vector<std::pair<std::string, T>>& kvs );
    template<typename T> static std::string    DescKV(                    const std::vector<std::pair<std::string, T>>& kvs );

    static void SetMetaKV_( std::string& meta, const VS& keys, const VS& vals );
    void        setMetaKV_( const VS& keys, const VS& vals );


    std::string descMeta() const ;

    static INT         GetFirstStampIndex_OLD(const std::vector<int64_t>& stamps, int64_t discount=200000 );  // 200k us, ie 0.2 s




    static NP* MakeMetaKVS_ranges( const std::string& meta_, const char* ranges_, std::ostream* ss=nullptr );
    static NP* MakeMetaKVS_ranges2(const std::string& meta_, const char* ranges_, std::ostream* ss=nullptr );

    static void Resolve_ranges( std::vector<std::string>& specs, const std::vector<std::string>& keys, const char* ranges_, std::ostream* ss=nullptr );
    static void TimeOrder_ranges( std::vector<int>& spec_order, const std::vector<std::string>& specs, const std::vector<std::string>& keys, const std::vector<int64_t>& tt, std::ostream* ss=nullptr );

    static NP*  MakeMetaKVS_ranges_table(
        const std::vector<int>& spec_order,
        const std::vector<std::string>& specs,
        const std::vector<std::string>& keys,
        const std::vector<int64_t>& tt,
        std::ostream* ss=nullptr ) ;

    static NP* MakeMetaKVS_ranges2_table(
        const std::vector<std::string>& specs,
        const std::vector<std::string>& keys,
        const std::vector<int64_t>& tt,
        std::ostream* ss=nullptr ) ;

    static NP* MakeMetaKVS_ranges(  const std::vector<std::string>& keys, const std::vector<int64_t>& tt, const char* ranges_, std::ostream* ss=nullptr );
    static NP* MakeMetaKVS_ranges2( const std::vector<std::string>& keys, const std::vector<int64_t>& tt, const char* ranges_, std::ostream* ss=nullptr );



    static std::string DescMetaKVS_kvs(      const std::vector<std::string>& keys, const std::vector<std::string>& vals, const std::vector<int64_t>& tt );
    static std::string DescMetaKVS_juncture( const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* juncture_ );
    static std::string DescMetaKVS_ranges(   const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* ranges_ ) ;
    static std::string DescMetaKVS_ranges2(  const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* ranges_ ) ;

    static std::string DescMetaKVS(const std::string& meta, const char* juncture = nullptr, const char* ranges=nullptr );
    std::string descMetaKVS(const char* juncture=nullptr, const char* ranges=nullptr) const ;






    static std::string DescMetaKV(const std::string& meta, const char* juncture = nullptr, const char* ranges=nullptr );
    std::string descMetaKV(const char* juncture=nullptr, const char* ranges=nullptr) const ;


    const char* get_lpath() const ;


    template<typename T> static INT DumpCompare( const NP* a, const NP* b, unsigned a_column, unsigned b_column, const T epsilon );
    static INT Memcmp( const NP* a, const NP* b );
    static bool SameData( const NP* a, const NP* b );

    static NP* Concatenate(const char* dir, const std::vector<std::string>& names);

    template<typename T>
    static NP* Concatenate(const std::vector<T*>& aa );  // template allows use with "NP" and "const NP"

    static NP* Combine(const std::vector<const NP*>& aa, bool annotate=true, const NP* parasite=nullptr );
    template<typename... Args> static NP* Combine_(Args ... aa);  // Combine_ellipsis


    static bool Exists(const char* base, const char* rel, const char* name);
    static bool Exists(const char* dir, const char* name);
    static bool Exists(const char* path);
    static bool ExistsSidecar( const char* path, const char* ext );


    static const char NODATA_PREFIX = '@' ;
    static bool IsNoData(const char* path);
    static const char* PathWithNoDataPrefix(const char* path);


    int load(const char* dir, const char* name);
    int load(const char* path);

    int load_string_(  const char* path, const char* ext, std::string& str );
    int load_strings_( const char* path, const char* ext, std::vector<std::string>* vstr );
    int load_meta(  const char* path );
    int load_names( const char* path );
    int load_labels( const char* path );

    void save_string_( const char* path, const char* ext, const std::string& str ) const ;
    void save_strings_(const char* path, const char* ext, const std::vector<std::string>& vstr ) const ;
    void save_meta( const char* path) const ;
    void save_names(const char* path) const ;
    void save_labels(const char* path) const ;

    void save_header(const char* path);
    void old_save(const char* path) ;  // formerly the *save* methods could not be const because of update_headers
    void save(const char* path) const ;  // *save* methods now can be const due to dynamic creation of header

    void save(const char* dir, const char* name) const ;
    void save(const char* dir, const char* reldir, const char* name) const ;

    void save_jsonhdr(const char* path) const ;
    void save_jsonhdr(const char* dir, const char* name) const ;

    std::string get_jsonhdr_path() const ; // .npy -> .npj on loaded path
    void save_jsonhdr() const ;

    template<typename T> std::string _present(T v) const ;
    template<typename T> void _dump(INT i0=-1, INT i1=-1, INT j0=-1, INT j1=-1) const ;


    template<typename T> void read(const T* src);
    template<typename T> void read2(const T* src);
    template<typename T> void write(T* dst) const ;


    template<typename T> static void Write(const char* dir, const char* name, const std::vector<T>& values );
    template<typename T> static void Write(const char* dir, const char* name, const T* data, INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 );
    template<typename T> static void Write(const char* dir, const char* reldir, const char* name, const T* data, INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 );
    template<typename T> static void Write(const char* path                 , const T* data, INT ni=-1, INT nj=-1, INT nk=-1, INT nl=-1, INT nm=-1, INT no=-1 );


    static void WriteNames(const char* dir, const char* name,                     const std::vector<std::string>& names, unsigned num_names=0, bool append=false );
    static void WriteNames(const char* dir, const char* reldir, const char* name, const std::vector<std::string>& names, unsigned num_names=0, bool append=false );
    static void WriteNames(const char* path,                                      const std::vector<std::string>& names, unsigned num_names=0, bool append=false );


    static void WriteNames_Simple( const char* dir, const char* name, const std::vector<std::string>& names );
    static void WriteNames_Simple( const char* path,                  const std::vector<std::string>& names );

    static void WriteString(const char* dir, const char* name, const char* ext, const std::string& str, bool append=false );

    static void ReadNames(const char* dir, const char* name, std::vector<std::string>& names ) ;
    static void ReadNames(const char* path,                  std::vector<std::string>& names ) ;


    template<typename T>
    static std::string DescKV(const std::vector<std::string>& keys, std::vector<T>& vals, std::vector<std::string>* extra);

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

    template <typename T>
    static NP* LoadFromTxtFile(const char* path);

    template <typename T>
    static NP* LoadFromTxtFile(const char* base, const char* relp);

    // FindUnit returns last matching unit string, so more specific strings that contain earlier
    // ones should come later in list
    static constexpr const char* UNITS = "eV MeV nm mm cm m ns g/cm2/MeV" ;
    static char* FindUnit(const char* line, const std::vector<std::string>& units  );
    static void Split(std::vector<std::string>& elems, const char* str, char delim);
    static void GetUnits(std::vector<std::string>& units );
    static bool IsListed(const std::vector<std::string>& ls, const char* str);
    static std::string StringConcat(const std::vector<std::string>& ls, char delim=' ' );

    template <typename T>
    static NP* ZEROProp(T dscale=1.);

    template <typename T>
    static NP* LoadFromString(const char* str, const char* path_for_debug_messages=nullptr );

    static unsigned CountChar(const char* str, char q );
    static void ReplaceCharInsitu(       char* str, char q, char n, bool first );
    static const char* ReplaceChar(const char* str, char q, char n, bool first );

    static const char* Resolve( const char* spec) ;
    static const char* ResolveProp(const char* spec);

    // END OF TAIL STATICS

    // primary data members
    std::vector<char> data = {} ;
    std::vector<INT>  shape ;
    std::string       meta ;
    std::vector<std::string>  names ;
    std::vector<std::string>* labels ;

    // non-persisted transients, set on loading
    std::string lpath ;
    std::string lfold ;

    // headers used for transport
    std::string _hdr ;
    std::string _prefix ;

    // results from parsing _hdr or set_dtype
    const char* dtype ;
    char        uifc ;    // element type code
    INT         ebyte ;   // element bytes
    INT         size ;    // number of elements from shape

    // nodata:true used for lightweight access to metadata from many arrays
    bool        nodata ;


};


//  SPECIALIZED MEMBER FUNCTIONS


template<typename T> inline const T*  NP::cvalues() const { return (T*)data.data() ;  }
template<typename T> inline T*        NP::values() { return (T*)data.data() ;  }

template<typename T> inline void NP::fill(T value)
{
    T* vv = values<T>();
    for(INT i=0 ; i < size ; i++) *(vv+i) = value ;
}

template<typename T> inline void NP::_fillIndexFlat(T offset)
{
    T* vv = values<T>();
    for(INT i=0 ; i < size ; i++) *(vv+i) = T(i) + offset ;
}


/**
BLOCK OF TEMPLATE SPECIALIZATIONS cvalues, values, _fillIndexFlat
-------------------------------------------------------------------

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


// STATIC CREATION METHODS

template<typename T>
inline NP* NP::MakeFromValues( const T* vals, INT num_vals )
{
    NP* a = NP::Make<T>(num_vals) ;
    T* aa = a->values<T>();
    for(INT i=0 ; i < num_vals ; i++) aa[i] = vals[i] ;
    return a ;
}

template <typename T>
inline NP::INT NP::ALength(T x0, T x1, T dx) // static
{
    T x = x0 ;
    INT n = 0 ;
    while( x < x1 )  // "<=" OR "<" ?  Follow np.arange
    {
       x += dx ;
       n++ ;
    }
    return n ;
}

/**
NP::ARange
-------------

This follows NumPy np.arange in not giving end values.
If you want to hit an end value use NP::Linspace.

::

    In [6]: a = np.arange(10,100,10,dtype=np.float32) ; a
    Out[6]: array([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=float32)

    In [7]: a.shape
    Out[7]: (9,)

**/


template<typename T>
inline NP* NP::ARange( T x0, T x1, T dx ) // static
{
    assert( x1 > x0 );
    assert( dx > 0. ) ;
    INT ni = ALength(x0,x1,dx) ;
    NP* a = NP::Make<T>(ni) ;
    T* aa = a->values<T>() ;
    for(INT i=0 ; i < ni ; i++ ) aa[i] = x0 + T(i)*dx ;
    return a ;
}


template <typename T>
inline NP* NP::Linspace( T x0, T x1, unsigned nx, INT npayload )  // static
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
NP::DeltaColumn
------------------

::

    In [6]: ab.a.stamps.shape
    Out[6]: (10, 13)

    In [7]: delta_stamps = ab.a.stamps - ab.a.stamps[:,0, np.newaxis]  ; delta_stamps
    Out[7]:
    array([[    0,   209,   223,   265,   265,   489,   505,   522,   723,  2097,  2097, 63816, 63919],
           [    0,   231,   244,   284,   284,   285,   368,   394,   590,   633,   633, 57248, 57356],
           [    0,   233,   245,   285,   285,   286,   351,   380,   638,   681,   681, 57402, 57480],
           [    0,   133,   170,   173,   173,   175,   259,   305,   844, 30887, 30888, 60904, 60961],
           [    0,   187,   226,   229,   230,   232,   396,   471,  1188, 33499, 33500, 63340, 63406],
           [    0,   170,   210,   214,   215,   217,   294,   328,   634, 31164, 31164, 60558, 60630],
           [    0,   131,   171,   174,   175,   177,   237,   273,   570, 32739, 32740, 62156, 62219],
           [    0,   136,   175,   179,   179,   181,   242,   292,   827, 32244, 32244, 62329, 62389],
           [    0,   135,   175,   179,   179,   181,   247,   281,   597, 32904, 32904, 62951, 63012],
           [    0,   132,   170,   174,   175,   177,   237,   271,   565, 32285, 32285, 62043, 62105]])

    In [8]: delta_stamps.shape
    Out[8]: (10, 13)

**/

template<typename T> inline NP* NP::DeltaColumn(const NP* a, INT jcol )
{
    assert( a->shape.size() == 2 );
    INT ni = a->shape[0] ;
    INT nj = a->shape[1] ;
    assert( jcol < nj );

    NP* b = NP::MakeLike(a) ;

    const T* aa = a->cvalues<T>();
    T* bb = b->values<T>();

    for(INT i=0 ; i < ni ; i++)
    for(INT j=0 ; j < nj ; j++)
    bb[i*nj+j] = aa[i*nj+j] - aa[i*nj+jcol] ;

    return b ;
}

/**
NP::ThetaRadians
-------------------

Angle range from zero to theta_max_pi

**/

template<typename T> inline NP* NP::ThetaRadians(INT nx, T theta_max_pi ) // static
{
    NP* a = NP::Make<T>(nx);
    T* aa = a->values<T>();
    for(INT i=0 ; i < nx ; i++)
    {
        T frac = nx == 1 ? T(0) : T(i)/T(nx-1) ;
        aa[i] = theta_max_pi*M_PI*frac ;
    }
    return a ;
}

/**
NP::MinusCosThetaLinearAngle
------------------------------

Returns array of nx values from -1 to 1 whwre the
spacing is calculated to make the steps linear
in the angle. For example with nx=181 the -cos(theta)
values will be provided at integer degrees from 0. to 180.

**/

template<typename T> inline NP* NP::MinusCosThetaLinearAngle(INT nx) // static
{
    NP* a = NP::Make<T>(nx);
    T* aa = a->values<T>();
    for(INT i=0 ; i < nx ; i++)
    {
        T frac = nx == 1 ? T(0) : T(i)/T(nx-1) ;
        T theta = frac*M_PI ;
        aa[i] = -cos(theta) ;
    }
    return a ;
}

inline NP* NP::SqrtOneMinusSquare( const NP* a ) // static
{
    assert( a->uifc == 'f' );
    assert( a->ebyte == 4 || a->ebyte == 8  );
    assert( a->shape.size() == 1 );
    INT num = a->shape[0] ;

    NP* b = NP::MakeLike(a);
    assert( b->ebyte == a->ebyte );

    if( a->ebyte == 8 )
    {
        const double* aa = a->cvalues<double>();
        double* bb = b->values<double>();
        for(INT i=0 ; i < num ; i++ ) bb[i] = sqrt(1.  - aa[i]*aa[i]) ;
    }
    else if( a->ebyte == 4 )
    {
        const float* aa = a->cvalues<float>();
        float* bb = b->values<float>();
        for(INT i=0 ; i < num ; i++ ) bb[i] = sqrt(1.f - aa[i]*aa[i]) ;
    }
    return b ;
}


inline NP* NP::Cos( const NP* a ) // static
{
    assert( a->uifc == 'f' );
    assert( a->ebyte == 4 || a->ebyte == 8  );
    assert( a->shape.size() == 1 );
    INT ni = a->shape[0] ;
    NP* b = NP::MakeLike(a);
    for(INT i=0 ; i < ni ; i++ )
    {
        int idx = i ;
        if( a->ebyte == 8 )
        {
            const double* aa = a->cvalues<double>();
            double* bb = b->values<double>();
            bb[idx] = std::cos(aa[idx]) ;
        }
        else if ( a->ebyte == 4 )
        {
            const float* aa = a->cvalues<float>();
            float* bb = b->values<float>();
            bb[idx] = std::cos(aa[idx]) ;
        }
    }
    return b ;

}

inline NP* NP::MakeWithCosineDomain( const NP* a, bool reverse ) // static
{
    assert( a->uifc == 'f' );
    assert( a->ebyte == 4 || a->ebyte == 8  );
    assert( a->shape.size() == 2 );
    INT ni = a->shape[0] ;
    INT nj = a->shape[1] ;
    assert( nj == 2 );

    NP* b = NP::MakeLike(a);
    assert( b->ebyte == a->ebyte );

    for(INT i=0 ; i < ni ; i++ )
    {
        INT a_item = i ;
        INT b_item = reverse ? ni - 1 - i : i ;

        for(INT j=0 ; j < nj ; j++ )
        {
            int a_idx = a_item*nj + j ;
            int b_idx = b_item*nj + j ;  ;

            if( a->ebyte == 8 )
            {
                const double* aa = a->cvalues<double>();
                double* bb = b->values<double>();
                bb[b_idx] = j == 0 ? std::cos(aa[a_idx]) : aa[a_idx] ;
            }
            else if ( a->ebyte == 4 )
            {
                const float* aa = a->cvalues<float>();
                float* bb = b->values<float>();
                bb[b_idx] = j == 0 ? std::cos(aa[a_idx]) : aa[a_idx] ;
            }
        }
    }

    return b ;
}





inline NP* NP::Incremented( const NP* a, INT offset ) // static
{
    assert( a->uifc == 'i' );
    assert( a->ebyte == 4 || a->ebyte == 8  );
    INT num = a->num_values() ;  // all dimensions

    NP* b = NP::MakeLike(a);

    if( a->ebyte == 8 )
    {
        const long* aa = a->cvalues<long>();
        long* bb = b->values<long>();
        for(INT i=0 ; i < num ; i++ ) bb[i] = aa[i] + long(offset) ;
    }
    else if( a->ebyte == 4 )
    {
        const int* aa = a->cvalues<int>();
        int* bb = b->values<int>();
        for(INT i=0 ; i < num ; i++ ) bb[i] = aa[i] + offset ;
    }
    return b ;
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

    INT dst_ni = dst_bins + 1 ;
    INT dst_nj = ndim == 2 ? src->shape[1] : -1 ;

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



template <typename T> NP* NP::Make( INT ni_, INT nj_, INT nk_, INT nl_, INT nm_, INT no_ ) // static
{
    std::string dtype = descr_<T>::dtype() ;
    NP* a = new NP(dtype.c_str(), ni_,nj_,nk_,nl_,nm_, no_) ;
    return a ;
}

template<typename T, typename ... Args> NP*  NP::Make_( Args ... shape_ )   // Make_shape static
{
    std::string dtype = descr_<T>::dtype() ;
    std::vector<INT> shape = {shape_ ...};
    NP* a = new NP(dtype.c_str(), shape ) ;
    return a ;
}

template<typename T> NP* NP::MakeFlat(INT ni, INT nj, INT nk, INT nl, INT nm, INT no ) // static
{
    NP* a = NP::Make<T>(ni, nj, nk, nl, nm, no );
    a->fillIndexFlat();
    return a ;
}




//  MEMBER FUNCTIONS


inline char*        NP::bytes() { return (char*)data.data() ;  }
inline const char*  NP::bytes() const { return (char*)data.data() ;  }

inline NP::INT NP::hdr_bytes() const { return _hdr.length() ; }
inline NP::INT NP::num_items() const { return shape[0] ;  }
inline NP::INT NP::num_values() const { return NPS::size(shape) ;  }
inline NP::INT NP::num_itemvalues() const { return NPS::itemsize(shape) ;  }
inline NP::INT NP::arr_bytes()  const { return NPS::size(shape)*ebyte ; }
inline NP::INT NP::item_bytes() const { return NPS::itemsize(shape)*ebyte ; }
inline NP::INT NP::meta_bytes() const { return meta.length() ; }


template<typename T>
inline bool NP::is_itemtype() const  // size of item matches size of type
{
    return item_bytes() == sizeof(T) ;
}

/**
NP::clear
----------

Note that std::vector::clear by itself does not deallocate
the memory, it is necessary in addition to call std::vector::shrink_to_fit
and even that is non-binding.

**/


inline void NP::clear()
{
    data.clear();
    data.shrink_to_fit();
    shape[0] = 0 ;
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

/**
NP::decode_header
-----------------------

Array header _hdr is parsed setting the below and data is resized.

shape
    vector of INT
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
    if(!nodata) data.resize(size*ebyte) ;   // data is now just char
    return true  ;
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




// CTOR
inline NP::NP(const char* dtype_, const std::vector<INT>& shape_ )
    :
    shape(shape_),
    labels(nullptr),
    dtype(strdup(dtype_)),
    uifc(NPU::_dtype_uifc(dtype)),
    ebyte(NPU::_dtype_ebyte(dtype)),
    size(NPS::size(shape)),
    nodata(false)
{
    init();
}

// DEFAULT CTOR
inline NP::NP(const char* dtype_, INT ni, INT nj, INT nk, INT nl, INT nm, INT no )
    :
    labels(nullptr),
    dtype(strdup(dtype_)),
    uifc(NPU::_dtype_uifc(dtype)),
    ebyte(NPU::_dtype_ebyte(dtype)),
    size(NPS::set_shape(shape, ni,nj,nk,nl,nm,no )),
    nodata(false)
{
    init();
}

inline void NP::init()
{
    unsigned long long size_ = size ;
    unsigned long long ebyte_ = ebyte ;
    unsigned long long num_char = size_*ebyte_ ;

    if(VERBOSE) std::cout
        << "NP::init"
        << " size " << size
        << " ebyte " << ebyte
        << " num_char " << num_char
        << std::endl
        ;

    data.resize( num_char ) ;  // vector of char
    std::fill( data.begin(), data.end(), 0 );
    _prefix.assign(net_hdr::LENGTH, '\0' );
    _hdr = make_header();
}




inline void NP::set_shape(INT ni, INT nj, INT nk, INT nl, INT nm, INT no)
{
    size = NPS::copy_shape(shape, ni, nj, nk, nl, nm, no);
    init();
}
inline void NP::set_shape(const std::vector<INT>& src_shape)
{
    size = NPS::copy_shape(shape, src_shape);
    init();
}

inline bool NP::has_shape(INT ni, INT nj, INT nk, INT nl, INT nm, INT no) const
{
    unsigned ndim = shape.size() ;
    return
           ( ni == -1 || ( ndim > 0 && INT(shape[0]) == ni)) &&
           ( nj == -1 || ( ndim > 1 && INT(shape[1]) == nj)) &&
           ( nk == -1 || ( ndim > 2 && INT(shape[2]) == nk)) &&
           ( nl == -1 || ( ndim > 3 && INT(shape[3]) == nl)) &&
           ( nm == -1 || ( ndim > 4 && INT(shape[4]) == nm)) &&
           ( no == -1 || ( ndim > 5 && INT(shape[5]) == no))
           ;
}


/**
NP::change_shape
------------------

One dimension can be -1 causing it to be filled automatically.
See tests/NPchange_shapeTest.cc

**/

inline void NP::change_shape(INT ni, INT nj, INT nk, INT nl, INT nm, INT no)
{
    INT size2 = NPS::change_shape(shape, ni, nj, nk, nl, nm, no);
    bool expect =  size == size2  ;
    if(!expect) std::raise(SIGINT) ;
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
        INT ni = 1 ;
        for(INT i=0 ; i < INT(ndim) - 2 ; i++) ni *= shape[i] ;
        // scrunch up the higher dimensions
        change_shape(ni, shape[ndim-2], shape[ndim-1] );
        if(VERBOSE) std::cerr << "NP::change_shape_to_3D : changed shape to : " << sstr() << std::endl  ;
    }
}

inline void NP::reshape( const std::vector<INT>& new_shape )
{
    NPS::reshape(shape, new_shape);
}

/**
NP::size_2D
-------------

Returns the conventional 2D (width, height) for payload last dimension P
passed by template variable. For example with an array of shape::

    (ni, nj, nk, nl, 4 )

A call to::

   a->size_2D<4>( width, height)

Would return::

   height = ni*nj*nk
   width = nl

NB the last dimension must match the template variable, 4 in the above example.

**/

template<int P>
inline void NP::size_2D( INT& width, INT& height ) const
{
    NPS::size_2D<P>(width, height, shape) ;
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
    INT  ebyte_ = NPU::_dtype_ebyte(dtype_) ;
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
        INT expand = ebyte/ebyte_ ;
        std::cout << "NP::set_dtype : shifting to smaller ebyte increases array dimensions, expand: " << expand << std::endl ;
        for(unsigned i=0 ; i < shape.size() ; i++ ) shape[i] *= expand ;
    }
    else if( ebyte_ > ebyte )
    {
        INT shrink = ebyte_/ebyte ;
        std::cout << "NP::set_dtype : shifting to larger ebyte decreases array dimensions, shrink: " << shrink << std::endl ;
        for(unsigned i=0 ; i < shape.size() ; i++ ) shape[i] /= shrink  ;
    }

    INT num_bytes  = size*ebyte ;      // old
    INT size_ = NPS::size(shape) ;     // new
    INT num_bytes_ = size_*ebyte_ ;    // new

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


/**
NP::index
-----------

Provides the flat value index from a set of integer dimension indices.
Negative dimension indices are interpreted to count from the back, ie -1 is the last element
in a dimension.

**/

inline NP::INT NP::index( INT i,  INT j,  INT k,  INT l, INT m, INT o ) const
{
    INT nd = shape.size() ;
    INT ni = nd > 0 ? shape[0] : 1 ;
    INT nj = nd > 1 ? shape[1] : 1 ;
    INT nk = nd > 2 ? shape[2] : 1 ;
    INT nl = nd > 3 ? shape[3] : 1 ;
    INT nm = nd > 4 ? shape[4] : 1 ;
    INT no = nd > 5 ? shape[5] : 1 ;

    INT ii = i < 0 ? ni + i : i ;
    INT jj = j < 0 ? nj + j : j ;
    INT kk = k < 0 ? nk + k : k ;
    INT ll = l < 0 ? nl + l : l ;
    INT mm = m < 0 ? nm + m : m ;
    INT oo = o < 0 ? no + o : o ;

    return  ii*nj*nk*nl*nm*no + jj*nk*nl*nm*no + kk*nl*nm*no + ll*nm*no + mm*no + oo ;
}

/**
NP::index0 : Provides element offset
---------------------------------------

Same as NP::index but -ve "missing" indices are treated as if they were zero.

**/

inline NP::INT NP::index0( INT i,  INT j,  INT k,  INT l, INT m, INT o) const
{
    INT nd = shape.size() ;

    INT ni = nd > 0 ? shape[0] : 1 ;
    INT nj = nd > 1 ? shape[1] : 1 ;
    INT nk = nd > 2 ? shape[2] : 1 ;
    INT nl = nd > 3 ? shape[3] : 1 ;
    INT nm = nd > 4 ? shape[4] : 1 ;
    INT no = nd > 5 ? shape[5] : 1 ;

    INT ii = i < 0 ? 0 : i ;
    INT jj = j < 0 ? 0 : j ;
    INT kk = k < 0 ? 0 : k ;
    INT ll = l < 0 ? 0 : l ;
    INT mm = m < 0 ? 0 : m ;
    INT oo = o < 0 ? 0 : o ;

    if(!(ii <  ni)) std::cerr << "NP::index0 ii/ni " << ii << "/" << ni  << std::endl ;

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

inline NP::INT NP::dimprod(unsigned q) const   // product of dimensions starting from dimension q
{
    INT dim = 1 ;
    for(INT d=q ; d < INT(shape.size()) ; d++) dim *= shape[d] ;
    return dim ;
}


template<typename... Args>
inline NP::INT NP::index_(Args ... idxx_) const
{
    std::vector<INT> idxx = {idxx_...};
    return index__(idxx);
}

template<typename... Args>
inline NP::INT NP::stride_(Args ... idxx_) const
{
    std::vector<INT> idxx = {idxx_...};
    return stride__(idxx);
}

template<typename... Args>
inline NP::INT NP::offset_(Args ... idxx_) const
{
    std::vector<INT> idxx = {idxx_...};
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
    std::vector<INT> idxx = {idxx_...};
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
   std::vector<INT> idxx = {idxx_...};
   slice_(out, idxx);
}




/**
NP::slice_
-----------

Collect a slice of values from the array into the out vector which is
first resized for fit them.

**/


template<typename T> inline void NP::slice_(std::vector<T>& out, const std::vector<INT>& idxx ) const
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

    INT slicedim = pickdim__(idxx);
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

inline std::string NP::DescIdx(const std::vector<INT>& idxx ) // static
{
    std::stringstream ss ;
    for(INT d=0 ; d < INT(idxx.size()) ; d++) ss << idxx[d] << " " ;
    std::string s = ss.str();
    return s ;
}


/**
NP::pickdim__
----------------

Returns ordinal of first -1 in idxx ?

**/

inline NP::INT NP::pickdim__(const std::vector<INT>& idxx) const
{
    INT pd = -1 ;
    INT num = 0 ;
    for(INT d=0 ; d < INT(shape.size()) ; d++)
    {
        INT dd = (d < INT(idxx.size()) ? idxx[d] : 1) ;
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

Flat value index obtained from array indices, a -ve index terminates
the summation over dimensions so only the dimensions to the left of the
-1 are summed.  This is used from NP::slice to give the start index
of the slice where the slice dimension is marked by the -1.

**/

inline NP::INT NP::index__(const std::vector<INT>& idxx) const
{
    INT idx = 0 ;
    for(INT d=0 ; d < INT(shape.size()) ; d++)
    {
        INT dd = (d < INT(idxx.size()) ? idxx[d] : 1) ;
        if( dd == -1 ) break ;
        idx += dd*dimprod(d+1) ;
    }
    return idx ;
}

/**
NP::stride__
--------------

1. find ordinal of first -1 in idxx
2. compute stride from product of array dimensions to the right of the -1

For example with an array of shape (100000, 32, 4, 4) and idxx (-1,)
the pickdim is 0 and the stride is 32*4*4

**/


inline NP::INT NP::stride__(const std::vector<INT>& idxx) const
{
    INT pd = pickdim__(idxx);
    assert( pd > -1 );
    INT stride = dimprod(pd+1) ;
    return stride ;
}


/**
NP::offset__
--------------

1. find ordinal of first -1 in idxx
2.


**/


inline NP::INT NP::offset__(const std::vector<INT>& idxx) const
{
    INT pd = pickdim__(idxx);
    assert( pd > -1 );

    INT offset = 0 ;
    for(INT d=pd+1 ; d < INT(shape.size()) ; d++)
    {
        INT dd = (d < INT(idxx.size()) ? idxx[d] : 1) ;
        offset += dd*dimprod(d+1) ;
    }
    return offset ;
}


inline NP::INT NP::itemsize_(INT i, INT j, INT k, INT l, INT m, INT o) const
{
    return NPS::itemsize_(shape, i, j, k, l, m, o) ;
}

inline void NP::itembytes_(const char** start,  INT& num_bytes,  INT i,  INT j,  INT k,  INT l, INT m, INT o ) const
{
    INT idx0 = index0(i,j,k,l,m,o) ;
    *start = bytes() + idx0*ebyte ;

    INT sz = itemsize_(i, j, k, l, m, o) ;
    num_bytes = sz*ebyte ;
}




template<typename T> inline T NP::get( INT i,  INT j,  INT k,  INT l, INT m, INT o) const
{
    unsigned idx = index(i, j, k, l, m, o);
    const T* vv = cvalues<T>() ;
    return vv[idx] ;
}

template<typename T> inline void NP::set( T val, INT i,  INT j,  INT k,  INT l, INT m, INT o)
{
    unsigned idx = index(i, j, k, l, m, o);
    T* vv = values<T>() ;
    vv[idx] = val ;
}



template<typename T> inline bool NP::is_allzero() const
{
    T zero = T(0) ;
    const T* vv = cvalues<T>();
    INT num = 0 ;
    for(INT i=0 ; i < size ; i++) if(vv[i] == zero) num += 1 ;
    bool allzero = num == size ;
    return allzero ;
}

inline bool NP::is_empty() const
{
    return shape.size() > 0 && shape[0] == 0 ;
}


/**
NP::descValues
----------------

NB the implicit double instanciation used by this method requires
this method to be implemented after the block of explicit cvalues specializations.

**/
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


inline std::string NP::descSize() const
{
    std::stringstream ss ;
    ss << "NP::descSize"
       << " arr_bytes " << arr_bytes()
       << " arr_kb " << arr_bytes()/1000
       ;
    std::string str = ss.str();
    return str ;
}



/**
NP::descTable
----------------

**/

template<typename T>
inline std::string NP::descTable(int wid) const
{
    return descTable_<T>(wid, labels, &names );
}



template<typename T>
inline T NP::findMinimumTimestamp() const
{
    const T* vv = cvalues<T>() ;

    T MAX = std::numeric_limits<T>::max();
    T t0 = MAX ;

    INT nv = num_values() ;
    for(INT i=0 ; i < nv ; i++)
    {
        T t = vv[i] ;
        if(!U::LooksLikeTimestamp<T>(t)) continue ;
        if( t < t0 ) t0 = t ;
    }
    return t0 == MAX ? 0 : t0 ;
}


/**
NP::descTable_
-----------------

**/

template<typename T>
inline std::string NP::descTable_(int wid,
    const std::vector<std::string>* column_labels,
    const std::vector<std::string>* row_labels
  ) const
{
    bool with_column_totals = true ;



    std::stringstream ss ;
    ss << "[NP::descTable_ " << sstr() << std::endl ;
    int ndim = shape.size() ;
    bool skip = ndim != 2 ;
    if(skip)
    {
        ss << " ERROR : UNEXPECTED SHAPE ndim " << ndim << std::endl ;
        ss << " column_labels " << std::endl ;
        if(column_labels) for(int i=0 ; i < int(column_labels->size()) ; i++) ss << (*column_labels)[i] << std::endl ;
        ss << " row_labels " << std::endl ;
        if(row_labels) for(int i=0 ; i < int(row_labels->size()) ; i++) ss << (*row_labels)[i] << std::endl ;
    }

    if(!skip)
    {
        const T* vv = cvalues<T>() ;
        T t0 = findMinimumTimestamp<T>() ;

        INT ni = shape[0] ;
        INT nj = shape[1] ;
        INT cwid = wid ;
        INT rwid = 2*wid ;

        std::vector<std::string> column_smry ;
        if(column_labels) U::Summarize( column_smry, column_labels, cwid );
        bool with_column_labels = int(column_smry.size()) == nj ;

        std::vector<std::string> row_smry ;
        if(row_labels) U::Summarize( row_smry, row_labels, rwid );
        bool with_row_labels = int(row_smry.size()) == ni ;


        if(with_column_labels) for(int j=0 ; j < nj ; j++) ss
            << U::Space( with_row_labels && j == 0  ? rwid+1 : 0 )
            << std::setw(cwid)
            << column_smry[j]
            << ( j < nj -1 ? " " : "\n" )
            ;

        std::vector<T> column_totals(nj,0);
        int num_timestamp = 0 ;

        for(int i=0 ; i < ni ; i++)
        {
            if(with_row_labels) ss << std::setw(rwid) << row_smry[i] << " " ;
            for(int j=0 ; j < nj ; j++)
            {
                T v = vv[i*nj+j] ;
                bool timestamp = U::LooksLikeTimestamp<T>(v) ;
                if(timestamp) num_timestamp += 1 ;
                T pv = timestamp ? v - t0 : v  ;

                column_totals[j] += pv ;

                if( timestamp )
                {
                    ss
                        << std::setw(cwid)
                        << std::fixed
                        << std::setprecision(6)
                        << double(pv)/1000000
                        ;
                }
                else
                {
                    ss
                        << std::setw(cwid)
                        << pv
                        ;
                }
                ss << ( j < nj-1 ? " " : "\n" ) ;

            }
        }

        ss << "num_timestamp " << num_timestamp << " auto-offset from t0 " << t0 << std::endl ;

        if(with_column_totals)
        {
            if(with_row_labels) ss << std::setw(rwid) << "TOTAL:" << " " ;
            for(int j=0 ; j < nj ; j++)
            {
                T v = column_totals[j] ;
                ss
                    << std::setw(cwid)
                    << v
                    << ( j < nj - 1 ? " " : "\n" )
                    ;
            }
        }

        if(with_column_labels)
        {
            for(int j=0 ; j < nj ; j++)
            {
                if( strcmp(column_smry[j].c_str(), (*column_labels)[j].c_str()) != 0) ss
                    << ( j == 0 ? "\n" : "" )
                    << std::setw(cwid)
                    << column_smry[j]
                    << " : "
                    << (*column_labels)[j]
                    << std::endl
                    ;
            }
        }

        if(with_row_labels)
        {
            for(int i=0 ; i < ni ; i++)
            {
                if( strcmp(row_smry[i].c_str(), (*row_labels)[i].c_str()) != 0) ss
                    << ( i == 0 ? "\n" : "" )
                    << std::setw(rwid)
                    << row_smry[i]
                    << " : "
                    << (*row_labels)[i]
                    << std::endl
                    ;
            }
        }
    }
    ss << "]NP::descTable_ " << sstr() << std::endl ;

    std::string str = ss.str();
    return str ;
}







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
    b->nodata = a->nodata ;
    if(a->labels) b->labels = new std::vector<std::string>( a->labels->begin(), a->labels->end() ) ;

    // pass along transient strings set on loading
    b->lpath = a->lpath ;
    b->lfold = a->lfold ;
}


inline void NP::set_preserve_last_column_integer_annotation()
{
    set_meta<INT>(Preserve_Last_Column_Integer_Annotation, 1 );
}
inline bool NP::is_preserve_last_column_integer_annotation() const
{
    return 1 == get_meta<INT>(Preserve_Last_Column_Integer_Annotation, 0) ;
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

    if(a->nodata == false)
    {
        memcpy( b->bytes(), a->bytes(), a->arr_bytes() );
    }
    unsigned nv = a->num_values();

    if(VERBOSE) std::cout
        << "NP::MakeCopy"
        << " a.dtype " << a->dtype
        << " b.dtype " << b->dtype
        << " a.nodata " << a->nodata
        << " b.nodata " << b->nodata
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
   std::vector<INT> items = {items_...};
   return MakeSelectCopy_(src, &items );
}

template NP* NP::MakeSelectCopy( const NP* , INT );
template NP* NP::MakeSelectCopy( const NP* , INT, INT );
template NP* NP::MakeSelectCopy( const NP* , INT, INT, INT );
template NP* NP::MakeSelectCopy( const NP* , INT, INT, INT, INT );

inline NP* NP::MakeSelectCopyE_(  const NP* src, const char* ekey, const char* fallback, char delim )
{
    std::vector<INT>* items = U::GetEnvVec<INT>(ekey, fallback, delim );
    return NP::MakeSelectCopy_( src, items )  ;
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const char* items_ )
{
    std::vector<INT>* items = U::MakeVec<INT>(items_);
    return NP::MakeSelectCopy_( src, items );
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const std::vector<INT>* items )
{
    return items ? MakeSelectCopy_(src, items->data(), INT(items->size()) ) : NP::MakeCopy(src) ;
}
inline NP* NP::MakeSelectCopy_(  const NP* src, const INT* items, INT num_items )
{
    assert( items );
    for(INT i=0 ; i < num_items ; i++) assert( items[i] < INT(src->shape[0]) );
    std::vector<INT> dst_shape(src->shape) ;
    dst_shape[0] = num_items ;
    NP* dst = new NP(src->dtype, dst_shape);
    assert( src->item_bytes() == dst->item_bytes() );
    unsigned size = src->item_bytes();
    for(INT i=0 ; i < num_items ; i++)
    {
        memcpy( dst->bytes() + i*size, src->bytes() + items[i]*size , size );
    }

    // format string idlist list of items and set into metadata
    std::stringstream ss ;
    for(INT i=0 ; i < num_items ; i++) ss << items[i] << ( i < num_items-1 ? "," : "" ) ;
    std::string idlist = ss.str() ;
    dst->set_meta<std::string>("idlist", idlist );
    // item indices become "id" when you use them to make a selection

    return dst ;
}

/**
NP::MakeSelection
--------------------

*sel* is an array of indices into the *src* array

**/


inline NP* NP::MakeSelection( const NP* src, const NP* sel )
{
    INT num_sel = sel->shape[0] ;

    assert( sel->uifc == 'i' && sel->ebyte == 8 );
    assert( sel->shape.size() == 1 );

    std::vector<INT> dst_shape(src->shape) ;
    dst_shape[0] = num_sel ;
    NP* dst = new NP(src->dtype, dst_shape);

    unsigned size = src->item_bytes();
    const int64_t* sel_vv = sel->cvalues<int64_t>();
    for(INT i=0 ; i < num_sel ; i++)
    {
        int64_t sel_v = sel_vv[i] ;
        memcpy( dst->bytes() + i*size, src->bytes() + sel_v*size,  size );
    }

    return dst ;
}


/**
NP::ParseSliceString
------------------------

Parse string of the below forms int vector of integers where ":"
is special cased to become -1::

     [:,0,0,0]
     [:,0,0,1]
     [:,0,0,2]

**/

inline int NP::ParseSliceString(std::vector<INT>& idxx, const char* _sli )
{
    size_t len = _sli ? strlen(_sli) : 0 ;
    if(len < 2) return 1 ;

    const char* o = strstr(_sli, "[");
    const char* c = strstr(_sli, "]");

    if(o == nullptr) return 2 ;
    if(c == nullptr) return 3 ;
    if(c - o <= 0 ) return 4 ;

    // copy starting from the char after the "[" up to the char before the "]"
    char* sli = strndup(o+1, c - o - 1 );
    //std::cout << "NP::ParseSliceString /" << sli << "/\n" ;

    char delim = ',' ;

    std::stringstream ss;
    ss.str(sli);
    std::string s;
    while (std::getline(ss, s, delim))
    {
        if(0 == strcmp(s.c_str(),":"))
        {
            idxx.push_back(-1);
        }
        else
        {
            std::istringstream iss(s);
            INT t ;
            iss >> t ;
            idxx.push_back(t) ;
        }
    }
    return 0 ;
}



/**
NP::MakeSliceSelection
------------------------

In [15]: a[a[:,0,0,0] < 0].shape
Out[15]: (514, 32, 4, 4)

In [16]: a[a[:,0,0,0] > 0].shape
Out[16]: (486, 32, 4, 4)

**/


template<typename T>
inline NP* NP::MakeSliceSelection( const NP* src, const char* _sel )
{
    NP* sel = src->makeWhereSelection<T>(_sel);
    return MakeSelection(src, sel) ;
}

/**
NP::makeWhereSelection
-------------------------

1. parse selection string like [:,0,0,2] < 0.5
   extracting the slice specification and cut

2. get the slice and apply the cut to yield
   a where array of indices for which the
   selection is true

**/

template<typename T>
inline NP* NP::makeWhereSelection( const char* _sel ) const
{
    std::vector<INT> idxx ;
    int rc = ParseSliceString(idxx, _sel);
    if(rc !=0 ) std::cout << "NP::makeWhereSelection FAIL to parse [" << ( _sel ? _sel : "-" ) << "]\n" ;
    if(rc !=0 ) return nullptr ;

    const char* gt = strstr(_sel, ">");
    const char* lt = strstr(_sel, "<");
    if( gt && lt ) return nullptr ;
    const char* pt = gt ? gt : lt ;
    std::istringstream iss(pt+1) ;

    T cut(0.f);
    iss >> cut ;
    //std::cout << " cut[" << cut << "]\n";

    std::vector<T> vals ;
    slice_(vals, idxx);

    std::vector<INT> where ;
    //std::cout << " vals.size " << vals.size() << "\n" ;
    for(INT i=0 ; i < INT(vals.size()) ; i++)
    {
        T value = vals[i] ;
        bool select = ( gt && value > cut ) || ( lt && value < cut );
        if(select) where.push_back(i);
    }

    return MakeFromValues<INT>(where.data(), where.size() ) ;
}






/**
NP::MakeItemCopy
------------------

Finds the index of a single item from the src array specified by (i,j,k,l,m,n)
and copies that item into the destination array.

**/

inline NP* NP::MakeItemCopy(  const NP* src, INT i, INT j, INT k, INT l, INT m, INT o )
{
    std::vector<INT> sub_shape ;
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
inline void NP::item_shape(std::vector<INT>& sub, INT i, INT j, INT k, INT l, INT m, INT o ) const
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

inline NP* NP::spawn_item(  INT i, INT j, INT k, INT l, INT m, INT o  ) const
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
        INT item = i ;
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







template<typename S>
inline NP::INT NP::count_if(std::function<bool(const S*)> predicate) const
{
    assert( is_itemtype<S>() );  // size of type same as item_bytes
    const S* vv = cvalues<S>();
    INT ni = num_items();
    INT count = 0 ;
    for(INT i=0 ; i < ni ; i++) if(predicate(vv+i)) count += 1 ;
    return count ;
}

template<typename T>
inline NP* NP::simple_copy_if(std::function<bool(const T*)> predicate ) const
{
    assert( is_itemtype<T>() );  // size of type same as item_bytes

    INT ni = num_items();
    INT si = count_if<T>(predicate) ;
    assert( si <= ni );

    const T* aa = cvalues<T>();

    NP* b = NP::Make<T>(si) ;
    T* bb = b->values<T>();

    INT _si = 0 ;
    for(INT i=0 ; i < ni ; i++)
    {
        if(predicate(aa+i))
        {
            memcpy( bb + _si,  aa+i , sizeof(T) );
            _si += 1 ;
        }
    }
    assert( si == _si );
    return b ;
}

/**
NP::copy_if
------------

S: compound type, eg int4, sphoton, etc..
T: atomic base type use for array, eg int, float, double

::

    NP* hit = photon->copy_if<float,sphoton>(predicate) ;

**/


template<typename T, typename S>
inline NP* NP::copy_if(std::function<bool(const S*)> predicate ) const
{
    assert( sizeof(S) >= sizeof(T) );
    INT ni = num_items();

    INT si = count_if<S>(predicate) ;
    INT sj = sizeof(S) / sizeof(T) ;


    assert( si <= ni );
    std::vector<INT> sh(shape) ;
    INT nd = sh.size();

    assert( nd > 0 );
    sh[0] = si ;

    INT itemcheck = 1 ;
    for(INT i=1 ; i < nd ; i++) itemcheck *= sh[i] ;

    bool sj_expect = itemcheck == sj ;
    if(!sj_expect) std::raise(SIGINT) ;
    assert( sj_expect );

    const S* aa = cvalues<S>();

    NP* b = NP::Make_<T>(sh) ;
    S* bb = b->values<S>();

    INT _si = 0 ;
    for(INT i=0 ; i < ni ; i++)
    {
        if(predicate(aa+i))
        {
            memcpy( bb + _si,  aa+i , sizeof(S) );
            _si += 1 ;
        }
    }
    assert( si == _si );
    return b ;
}


/**
NP::flexible_copy_if
----------------------

S: compound type, eg int4, sphoton, etc..
T: atomic base type use for array, eg int, float, double
Args: variable number of ints used to specify item shape eg (4,4)

If no itemshape is provided used default of (sizeof(S)/sizeof(T),)
For example with sphoton that has size of 16 floats, would use::

    NP* hit = photon->copy_if<float,sphoton>(predicate, 4, 4) ;

HMM: as the source array item shape is already available there
there is actually no need for the Args itemshape complication.
Hence named this "flexible"
**/

template<typename T, typename S, typename... Args>
inline NP* NP::flexible_copy_if(std::function<bool(const S*)> predicate, Args ... itemshape ) const
{
    assert( sizeof(S) >= sizeof(T) );
    INT ni = num_items();

    INT si = count_if<S>(predicate) ;
    INT sj = sizeof(S) / sizeof(T) ;

    assert( si <= ni );

    std::vector<INT> itemshape_ = {itemshape...};
    std::vector<INT> sh ;
    sh.push_back(si) ;

    if(itemshape_.size() == 0 )
    {
        sh.push_back(sj) ;
    }
    else
    {
        INT itemcheck = 1 ;
        for(INT i=0 ; i < INT(itemshape_.size()) ; i++)
        {
            sh.push_back(itemshape_[i]) ;
            itemcheck *= itemshape_[i] ;
        }
        assert( itemcheck == sj );
    }
    const S* aa = cvalues<S>();

    NP* b = NP::Make_<T>(sh) ;
    S* bb = b->values<S>();

    INT _si = 0 ;
    for(INT i=0 ; i < ni ; i++)
    {
        if(predicate(aa+i))
        {
            memcpy( bb + _si,  aa+i , sizeof(S) );
            _si += 1 ;
        }
    }
    assert( si == _si );
    return b ;
}












inline NP* NP::LoadIfExists(const char* path_)
{
    return NP::Exists(path_) ? NP::Load(path_) : nullptr ;
}


inline NP* NP::Load(const char* path_)
{
    const char* path = U::Resolve(path_);
    if(VERBOSE)
        std::cerr
            << "[ NP::Load "
            << " path_ [" << ( path_  ? path_ : "-" ) << "]"
            << " path [" << ( path ? path : "-" ) << "]"
            << " int(strlen(path)) " << ( path ? int(strlen(path)) : -1 )
            << std::endl
            ;

    if(path == nullptr) return nullptr ; // eg when path_ starts with unsetenvvar "$TOKEN"

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


template<typename T>
inline NP* NP::LoadSlice( const char* _path, const char* _sel )
{
    const char* path = U::Resolve(_path);
    if(!Exists(path)) return nullptr ;

    NP* __a = NP::Load(path) ;
    NP* _a = __a ? NP::MakeNarrowIfWide(__a) : nullptr  ;

    const char* sel = nullptr ;
    if(_sel && strlen(_sel ) > 1)
    {
        bool starts_with_dollar = _sel[0] == '$' ;
        sel = starts_with_dollar ? U::GetEnv(_sel+1, nullptr) : _sel ;
        if(VERBOSE) std::cout
            << "NP::LoadSlice"
            << " starts_with_dollar " << ( starts_with_dollar ? "YES" : "NO " )
            << " _sel [" << ( _sel ? _sel : "-" ) << "]"
            << " sel ["  << ( sel ? sel : "-" ) << "]"
            << "\n"
            ;
    }
    NP* a = sel ? NP::MakeSliceSelection<T>(_a, sel) : _a ;
    return a ;
}




/**
NP::ExistsArrayFolder
----------------------

Returns true if the path does not end in .npy
and is a directory folder containing one or more .npy files.

**/

inline bool NP::ExistsArrayFolder(const char* path )
{
    bool npy_ext = U::EndsWith(path, EXT) ;
    if(npy_ext) return false ;
    std::vector<std::string> nms ;
    U::DirList(nms, path, EXT);
    return nms.size() > 0 ;
}





inline NP* NP::Load_(const char* path)
{
    if(!path) return nullptr ;
    NP* a = new NP() ;
    INT rc = a->load(path) ;
    return rc == 0 ? a  : nullptr ;
}

inline NP* NP::Load(const char* dir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, name);
    return Load(path.c_str());
}

inline NP* NP::Load(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, reldir, name);
    return Load(path.c_str());
}

/**
NP::LoadWide
--------------

Loads array and widens it to 8 bytes per element if not already wide.

**/

inline NP* NP::LoadWide(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, reldir, name);
    return LoadWide(path.c_str());
}

inline NP* NP::LoadWide(const char* dir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, name);
    return LoadWide(path.c_str());
}

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

/**
NP::LoadNarrow
---------------

Loads array and narrows to 4 bytes per element if not already narrow.

**/

inline NP* NP::LoadNarrow(const char* dir, const char* reldir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, reldir, name);
    return LoadNarrow(path.c_str());
}
inline NP* NP::LoadNarrow(const char* dir, const char* name)
{
    if(!dir) return nullptr ;
    std::string path = U::form_path(dir, name);
    return LoadNarrow(path.c_str());
}
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

/**
NP::find_value_index
---------------------

**/

template<typename T> inline NP::INT NP::find_value_index(T value, T epsilon) const
{
    const T* vv = cvalues<T>();
    unsigned ni = shape[0] ;
    unsigned ndim = shape.size() ;
    INT idx = -1 ;
    if(ndim == 1)
    {
        for(unsigned i=0 ; i < ni ; i++)
        {
            T v = vv[i];
            if(std::abs(v-value) < epsilon)
            {
                idx = i ;
                break ;
            }
        }
    }
    return idx ;
}

/**
NP::ifind2D
------------

Consider a 2D array of integers of shape (ni, nj).
Look for *ival* in the *jcol* column of each of the *ni* items
and return the corresponding *vret* from the *jret* column
or the *i* index if *jret* is -1.

::

    In [2]: a
    Out[2]:
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31],
           [32, 33, 34, 35],
           [36, 37, 38, 39]], dtype=int32)

::

    NP* a = NP::Make<int>(10,4) ;
    a->fillIndexFlat();
    a->save("/tmp/a.npy");

    int ival = 4 ;  // value to look for
    int jcol = 0 ;  // column in which to look for ival
    int jret = 3 ;  // column to return -> 7

    int vret = a->ifind2D<int>(ival, jcol, jret );

    std::cout << " vret " << vret << std::endl ;
    assert( vret == 7 );

**/

template<typename T> inline T NP::ifind2D(T ivalue, INT jcol, INT jret ) const
{
    if( shape.size() != 2 ) return -2 ;

    INT ni = shape[0] ;
    INT nj = shape[1] ;

    if( jcol >= nj ) return -3 ;
    if( jret >= nj ) return -4 ;

    const T* vv = cvalues<T>();

    T vret = -1 ;

    for(INT i=0 ; i < ni ; i++)
    {
        T vcol = vv[i*nj+jcol];
        bool match = vcol == ivalue ;

        T cand = jret < 0 ? i : vv[i*nj+jret];

        if(VERBOSE) std::cout
           << "NP::ifind2D"
           << " i " << i
           << " vcol " << vcol
           << " cand " << cand
           << " match " << match
           << std::endl
           ;

        if(match)
        {
            vret = cand ;
            break ;
        }
    }
    return vret ;
}




inline bool NP::is_pshaped() const
{
    bool property_shaped = shape.size() == 2 && shape[1] == 2 && shape[0] > 1 ;
    return property_shaped ;
}

template<typename T>
inline bool NP::is_pconst() const
{
    if(!is_pshaped()) return false ;
    const T* vv = cvalues<T>();
    INT ni = shape[0] ;
    INT nj = shape[1] ;
    const T v0 = vv[0*nj+nj-1] ;
    INT num_equal = 0 ;
    for(INT i=0 ; i < ni ; i++) num_equal += vv[i*nj+nj-1] == v0 ? 1 : 0 ;
    return num_equal == ni ;
}

/**
NP::is_pconst_dumb
-------------------

A dumb property is one that uses more than two energy points
to represent a constant value.

**/
template<typename T>
inline bool NP::is_pconst_dumb() const
{
    return is_pconst<T>() && shape[0] > 2 ;
}


template<typename T>
inline T NP::pconst(T fallback) const
{
    if(!is_pconst<T>()) return fallback ;
    INT nj = shape[1] ;
    const T* vv = cvalues<T>();
    const T v0 = vv[0*nj+nj-1] ;
    return v0 ;
}

template<typename T>
inline NP* NP::MakePCopyNotDumb(const NP* a) // static
{
    assert( a && a->is_pshaped() );
    NP* r = nullptr ;

    if(a->is_pconst_dumb<T>())
    {
        T dl = a->plhs<T>(0) ;
        T dr = a->prhs<T>(0) ;
        T vc = a->pconst<T>(-1) ;
        r = MakePConst<T>(dl, dr, vc );
    }
    else
    {
        r = MakeCopy(a);
    }
    return r ;
}

template<typename T>
inline NP* NP::MakePConst( T dl, T dr, T vc ) // static
{
    INT ni = 2 ;
    INT nj = 2 ;

    NP* p = NP::Make<T>(ni,nj) ;
    T*  pp = p->values<T>() ;

    pp[0*nj+0] = dl ;
    pp[0*nj+1] = vc ;
    pp[1*nj+0] = dr ;
    pp[1*nj+1] = vc ;

    return p ;
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

template<typename T> inline NP::INT  NP::pfindbin(const T value, unsigned column, bool& in_range) const
{
    const T* vv = cvalues<T>();

    unsigned ndim = shape.size() ;
    unsigned ni = shape[0] ;
    unsigned nj = ndim == 1 ? 1 : shape[1] ;
    assert( column < nj );

    const T lhs = vv[nj*(0)+column] ;
    const T rhs = vv[nj*(ni-1)+column] ;

    INT ibin = -1 ;
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

template<typename T> inline void  NP::get_edges(T& lo, T& hi, unsigned column, INT ibin) const
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
    else if( ibin == INT(ni) )
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

template<typename T> inline void NP::minmax(T& mn, T&mx, unsigned j, INT item ) const
{
    unsigned ndim = shape.size() ;
    assert( ndim == 2 || ndim == 3);

    unsigned ni = shape[ndim-2] ;
    unsigned nj = shape[ndim-1] ;
    assert( j < nj );

    unsigned num_items = ndim == 3 ? shape[0] : 1 ;
    assert( item < INT(num_items) );
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
NP::minmax2D_reshaped<N,T>
--------------------------

1. Temporarily change shape to (-1,N) : ie array of items with N element of type T
2. invoked minmax2D determining value range of the items
3. return the shape back to the original

Consider array of shape (1000,32,4,4) with (position,time) in [:,:,0]
After reshaping that becomes (1000*32*4, 4 )
BUT only every fourth 4-plet is (position, time)

So (item_stride, item_offset) needs to be (4,0) where the
item is the 4-plet chosen with the N template parameter.

**/
template<int N, typename T> inline void NP::minmax2D_reshaped(T* mn, T* mx, INT item_stride, INT item_offset )
{
    std::vector<INT> sh = shape ;
    change_shape(-1,N);

    assert( shape.size() == 2 );
    [[maybe_unused]] INT ni = shape[0] ;
    [[maybe_unused]] INT nj = shape[1] ;
    assert( nj == N && ni > 0 );

    minmax2D<T>(mn, mx, item_stride, item_offset );

    reshape(sh);
}

/**
NP::minmax2D
-------------

Assuming shape (-1, N) where N is typically small (eg 4)
and the mn, mx arguments point to structures
with at least N elements.

**/

template<typename T> inline void NP::minmax2D(T* mn, T* mx, INT item_stride, INT item_offset ) const
{
    assert( shape.size() == 2 );
    INT ni = shape[0] ;
    INT nj = shape[1] ;

    for(INT j=0 ; j < nj ; j++) mn[j] = std::numeric_limits<T>::max() ;
    for(INT j=0 ; j < nj ; j++) mx[j] = std::numeric_limits<T>::min() ;

    const T* vv = cvalues<T>() ;
    for(INT i=0 ; i < ni ; i++)
    {
        if( i % item_stride != item_offset ) continue ;
        for(INT j=0 ; j < nj ; j++)
        {
            INT idx = i*nj + j ;
            if( vv[idx] < mn[j] ) mn[j] = vv[idx] ;
            if( vv[idx] > mx[j] ) mx[j] = vv[idx] ;
        }
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

template<typename T> inline T  NP::pdomain(const T value, INT item, bool dump ) const
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
    assert( item < INT(num_items) );
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
template<typename T> inline T  NP::interp2D(T x, T y, INT item) const
{
    INT ndim = shape.size() ;
    assert( ndim == 2 || ndim == 3 );

    INT num_items = ndim == 3 ? shape[0] : 1 ;
    assert( item < num_items );
    INT ni = shape[ndim-2];
    INT nj = shape[ndim-1];  // typically 2, but can be more
    INT item_offset = item == -1 ? 0 : ni*nj*item ;   // item=-1 same as item=0

    const T* vv = cvalues<T>() + item_offset ;

    T xB = x - T(0.5) ;
    T yB = y - T(0.5) ;
    // decompose floating point value into integral and fractional parts
    T xBint ;
    T xBfra = std::modf(xB, &xBint);
    INT j = INT(xBint) ;

    T yBint ;
    T yBfra = std::modf(yB, &yBint);
    INT i = INT(yBint) ;

    const T one(1.);

#ifdef VERBOSE
    std::cout
        << " ni = " << ni
        << " nj = " << nj
        << " i = "  << i
        << " j = "  << j
        << std::endl
       ;
#endif

    bool i_inrange = i < ni && i > -1 ;
    bool j_inrange = j < nj && j > -1 ;
    bool ij_inrange = i_inrange && j_inrange ;

    if(!ij_inrange ) std::cerr
       << "NP::interp2D"
       << "\n"
       << " x " << std::fixed << std::setw(10) << std::setprecision(5) << x
       << " xB " << std::fixed << std::setw(10) << std::setprecision(5) << xB
       << " xBint " << std::fixed << std::setw(10) << std::setprecision(5) << xBint
       << " xBfra " << std::fixed << std::setw(10) << std::setprecision(5) << xBfra
       << " j " << j
       << " nj " << nj
       << "\n"
       << " y " << std::fixed << std::setw(10) << std::setprecision(5) << y
       << " yB " << std::fixed << std::setw(10) << std::setprecision(5) << yB
       << " yBint " << std::fixed << std::setw(10) << std::setprecision(5) << yBint
       << " yBfra " << std::fixed << std::setw(10) << std::setprecision(5) << yBfra
       << " i " << i
       << " ni " << ni
       << "\n"
       << " item " << item
       << " ndim " << ndim
       << " item_offset " << item_offset
       << " num_items " << num_items
       << " i_inrange " << ( i_inrange ? "YES" : "NO " )
       << " j_inrange " << ( j_inrange ? "YES" : "NO " )
       << " ij_inrange " << ( ij_inrange ? "YES" : "NO " )
       << "\n"
       ;

    assert( ij_inrange );




    // (i,j) => (y,x)
    T v00 = ij_inrange ? vv[(i+0)*nj+(j+0)] : 0. ;
    T v01 = ij_inrange ? vv[(i+0)*nj+(j+1)] : 0. ;   // v01 at j+1 (at large x than v00)
    T v10 = ij_inrange ? vv[(i+1)*nj+(j+0)] : 0. ;
    T v11 = ij_inrange ? vv[(i+1)*nj+(j+1)] : 0. ;

#ifdef VERBOSE
    std::cout
       << "NP::interp2D[ "
       << " T[ij] = " << v00
       << " T[i+1,j] = "<< v10
       << " T[i,j+1] = "<< v01
       << " T[i+1,j+1] = "<< v11
       << " NP::interp2D]"
       << std::endl
       ;
#endif

    // v10 is i+1

    // tex(x,y)=(1?¦Á(1?¦ÂT[i,j]+¦Á1?¦ÂT[i+1,j]+(1?¦Á¦Â[i,j+1]+¦fÂ[i+1,j+1]
    // hmm does this need a y-flip ?

    T z =  (one - xBfra)*(one - yBfra)*v00 +
                  xBfra *(one - yBfra)*v01 +
           (one - xBfra)*       yBfra *v10 +
                  xBfra *       yBfra *v11 ;

    return z ;
}


/**
NP::interp
------------

CAUTION: using the wrong type here somehow scrambles the array contents,
so always explicitly define the template type : DO NOT RELY ON COMPILER WORKING IT OUT.

**/

template<typename T> inline T NP::interp(T x, INT item) const
{
    unsigned ndim = shape.size() ;
    assert( ndim == 2 || ndim == 3 );

    unsigned num_items = ndim == 3 ? shape[0] : 1 ;
    bool num_items_expect = item < INT(num_items)  ;
    assert( num_items_expect );
    if(!num_items_expect) std::raise(SIGINT);

    unsigned ni = shape[ndim-2];
    unsigned nj = shape[ndim-1];  // typically 2 but can be more
    unsigned item_offset = item == -1 ? 0 : ni*nj*item ;   // item=-1 same as item=0

    assert( ni > 1 );
    assert( nj <= 8 );        // not needed for below, just for sanity of payload
    unsigned jdom = 0 ;       // 1st payload slot is "domain"
    unsigned jval = nj - 1 ;  // last payload slot is "value"   : TODO: argument to control this
    // note that with nj > 2 this allows other values to be carried

    const T* vv = cvalues<T>() + item_offset ;

    INT lo = 0 ;
    INT hi = ni-1 ;         // domain must be in ascending order

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
        INT mi = (lo+hi)/2;
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

template<typename T> inline T NP::interpHD(T u, unsigned hd_factor, INT item) const
{
    unsigned ndim = shape.size() ;
    assert( ndim == 3 || ndim == 4 );

    unsigned num_items = ndim == 4 ? shape[0] : 1 ;
    assert( item < INT(num_items) );

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
    INT lo = 0 ;
    INT hi = ni-1 ;

    if( u <= vv[lo*nj*nk+j*nk+kdom] ) return vv[lo*nj*nk+j*nk+kval] ;
    if( u >= vv[hi*nj*nk+j*nk+kdom] ) return vv[hi*nj*nk+j*nk+kval] ;

    // binary search for domain bin containing x
    while (lo < hi-1)
    {
        INT mi = (lo+hi)/2;
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

template<typename T> inline T NP::interp(INT i, T x) const
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

template<typename T> inline T NP::combined_interp_3(INT i, T x) const
{
    INT ndim = shape.size() ;
    assert( ndim == 3 && shape[ndim-1] >= 2 && i < shape[0] && shape[1] > 1 );

    INT stride = shape[ndim-2]*shape[ndim-1] ;
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

template<typename T> inline T NP::combined_interp_5(INT i, INT j, INT k, T x) const
{
    INT ndim = shape.size() ;
    assert( ndim == 5 );
    INT ni = shape[0] ;
    INT nj = shape[1] ;
    INT nk = shape[2] ;
    bool args_expect =  i < ni && j < nj && k < nk ;
    assert( args_expect );
    if(!args_expect) std::raise(SIGINT);

    INT nl = shape[ndim-2] ;
    INT nm = shape[ndim-1] ;

    bool shape_expect = nl > 1 && nm == 2  ;
    // require more than one domain items
    assert( shape_expect );
    if(!shape_expect) std::raise(SIGINT);

    INT stride = shape[ndim-2]*shape[ndim-1] ;
    INT iprop = i*nj*nk+j*nk+k ;
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

template<typename T> inline T NP::_combined_interp(const T* vv, INT niv, T x) const
{
    INT ndim = shape.size() ;
    INT ni = nview::int_from<T>( *(vv+niv-1) ) ; // NPU.hh:nview
    INT nj = shape[ndim-1] ;  // normally 2 with (dom, val)

    INT jdom = 0 ;       // 1st payload slot is "domain"
    INT jval = nj - 1 ;  // last payload slot is "value", with nj 2 (typical) that is 1

    INT lo = 0 ;
    INT hi = ni-1 ;

    if( x <= vv[nj*lo+jdom] ) return vv[nj*lo+jval] ;
    if( x >= vv[nj*hi+jdom] ) return vv[nj*hi+jval] ;

    while (lo < hi-1)
    {
        INT mi = (lo+hi)/2;
        if (x < vv[nj*mi+jdom]) hi = mi ;
        else lo = mi;
    }

    T dy = vv[nj*hi+jval] - vv[nj*lo+jval] ;
    T dx = vv[nj*hi+jdom] - vv[nj*lo+jdom] ;
    T y =  vv[nj*lo+jval] + dy*(x-vv[nj*lo+jdom])/dx ;
    return y ;
}

/**
NP::FractionalRange
---------------------

Return fraction of x within range x0 to x1 or 0 below and 1 above the range.

+-------------------+-------------+
| x <= x0           | T(0)        |
+-------------------+-------------+
| x >= x1           | T(1)        |
+-------------------+-------------+
| x0 < x < x1       | T(0->1)     |
+-------------------+-------------+

**/

template<typename T> inline T NP::FractionalRange( T x, T x0, T x1 )  // static
{
    assert( x1 > x0 );
    if( x <= x0 ) return T(0) ;
    if( x >= x1 ) return T(1) ;
    T xf = (x-x0)/(x1-x0) ;
    return xf ;
}



template<typename T> inline NP* NP::cumsum(INT axis) const
{
    assert( axis == 1 && "for now only axis=1 implemented" );
    const T* vv = cvalues<T>();
    NP* cs = NP::MakeLike(this) ;
    T* ss = cs->values<T>();
    for(INT p=0 ; p < size ; p++) ss[p] = vv[p] ;   // flat copy

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


inline void NP::dump(INT i0, INT i1, INT j0, INT j1) const
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




inline std::string NP::Brief(const NP* a)
{
    return a ? a->sstr() : "-" ;
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
    if(nodata) ss << " NODATA " ;
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


template<typename T>
inline std::string NP::repr() const
{
    const T* vv = cvalues<T>();
    INT nv = num_values() ;
    const INT edge = 5 ;

    std::stringstream ss ;
    ss << "{" ;
    for(INT i=0 ; i < nv ; i++)
    {
        if( i < edge || i > nv - edge )
        {
            switch(uifc)
            {
                case 'f': ss << std::setw(10) << std::fixed << std::setprecision(5) << vv[i] << " " ; break ;
                case 'u': ss << std::setw(5) << vv[i] << " " ; break ;
                case 'i': ss << std::setw(5) << vv[i] << " " ; break ;
                case 'c': ss << std::setw(10) << vv[i] << " " ; break ;   // TODO: check array of std::complex
            }
        }
        else if( i == edge )
        {
            ss << "... " ;
        }
    }
    ss << "}" ;
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
inline NP::INT NP::get_name_index( const char* qname ) const
{
    unsigned count = 0 ;
    return NameIndex(qname, count, names);
}
inline NP::INT NP::get_name_index( const char* qname, unsigned& count ) const
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

inline NP::INT NP::NameIndex( const char* qname, unsigned& count, const std::vector<std::string>& names ) // static
{
    if(names.size() == 0) return -1 ;

    INT result(-1);
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
    //return int(shape.size()) == 2 && shape[1] == 1 && shape[0] == int(names.size()) ;
    return INT(shape.size()) > 0 && shape[0] == INT(names.size()) ;
}

template<typename T>
inline T NP::get_named_value( const char* qname, T fallback ) const
{
    bool is_named = is_named_shape() ;

    if(NP::VERBOSE) std::cerr
        << "NP::get_named_value [" << qname << "]"
        << " is_named " << is_named
        << " sstr " << sstr()
        << std::endl
        ;

    if(! is_named) return fallback ;

    const T* vv = cvalues<T>() ;

    unsigned count(0);
    INT idx = get_name_index(qname, count );

    if(count != 1) return fallback ;
    if(idx < INT(shape[0])) return vv[idx] ;
    return fallback ;
}





inline bool NP::has_meta() const
{
    return meta.empty() == false ;
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

inline std::string NP::get_meta_string(const std::string& meta, const char* key)  // static
{
    const char* metadata = meta.empty() ? nullptr : meta.c_str() ;
    return get_meta_string_( metadata, key );
}

/**
NP::makeMetaKVProfileArray
----------------------------

1. finds metadata lines looking like profile stamps with keys containing the ptn,
   a nullptr ptn matches all lines
2. create (N,3) int64_t array filled with the stamps (t[us],vm[kb],rs[kb])

**/

inline NP* NP::makeMetaKVProfileArray(const char* ptn) const
{
    std::vector<std::string> keys ;
    std::vector<std::string> vals ;
    bool only_with_profile = true ;
    GetMetaKV(meta, &keys, &vals, only_with_profile, ptn );
    assert( keys.size() == vals.size() );
    INT num_key = keys.size();

    INT ni = num_key ;
    INT nj = 3 ;
    bool dump = false ;

    NP* prof = ni > 0 ? NP::Make<int64_t>(ni, nj ) : nullptr  ;
    int64_t* pp = prof ? prof->values<int64_t>() : nullptr ;
    if(prof) prof->labels = new std::vector<std::string> {"st[us]", "vm[kb]", "rs[kb]" } ;
    for(INT i=0 ; i < ni ; i++)
    {
        const char* k = keys[i].c_str();
        const char* v = vals[i].c_str();
        bool looks_like_prof  = U::LooksLikeProfileTriplet(v);
        assert( looks_like_prof );
        if(!looks_like_prof) continue ;

        char* end = nullptr ;
        int64_t st = strtoll( v,   &end, 10 ) ;
        int64_t vm = strtoll( end+1, &end , 10 ) ;
        int64_t rs = strtoll( end+1, &end , 10 ) ;

        if(dump) std::cout
            << "NP::makeMetaKVProfileArray"
            << " k " << ( k ? k : "-" )
            << " v " << ( v ? v : "-" )
            << " st " << st
            << " vm " << vm
            << " rs " << rs
            << std::endl
            ;

        pp[nj*i + 0 ] = st ;
        pp[nj*i + 1 ] = vm ;
        pp[nj*i + 2 ] = rs ;
        prof->names.push_back(k) ;
    }
    prof->meta = meta ;
    return prof ;
}

inline void NP::GetMetaKV_(
    const char* metadata,
    std::vector<std::string>* keys,
    std::vector<std::string>* vals,
    bool only_with_profile,
    const char* ptn
    ) // static
{
    if(metadata == nullptr) return ;
    std::stringstream ss;
    ss.str(metadata);
    std::string s;
    char delim = ':' ;

    while (std::getline(ss, s))
    {
        size_t pos = s.find(delim);
        if( pos != std::string::npos )
        {
            std::string _k = s.substr(0, pos);
            std::string _v = s.substr(pos+1);
            const char* k = _k.c_str();
            const char* v = _v.c_str();
            bool match_ptn = ptn ? strstr( k, ptn ) != nullptr : true  ;
            bool looks_like_profile = U::LooksLikeProfileTriplet(v);
            bool select = only_with_profile ? looks_like_profile && match_ptn : match_ptn ;
            if(!select) continue ;

            if(keys) keys->push_back(k);
            if(vals) vals->push_back(v);
        }
    }
}

inline void NP::GetMetaKV(
    const std::string& meta,
    std::vector<std::string>* keys,
    std::vector<std::string>* vals,
    bool only_with_profile,
    const char* ptn)  // static
{
    const char* metadata = meta.empty() ? nullptr : meta.c_str() ;
    return GetMetaKV_( metadata, keys, vals, only_with_profile, ptn  );
}




template<typename T> inline T NP::GetMeta(const std::string& mt, const char* key, T fallback) // static
{
    if(mt.empty()) return fallback ;
    std::string s = get_meta_string( mt, key);
#ifdef DEBUG
    std::cout << "NP::GetMeta[" << s << "]" << std::endl ;
#endif
    if(s.empty()) return fallback ;
    return U::To<T>(s.c_str()) ;
}


template int         NP::GetMeta<int>(        const std::string& , const char*, int ) ;
template unsigned    NP::GetMeta<unsigned>(   const std::string& , const char*, unsigned ) ;
template float       NP::GetMeta<float>(      const std::string& , const char*, float ) ;
template double      NP::GetMeta<double>(     const std::string& , const char*, double ) ;
template std::string NP::GetMeta<std::string>(const std::string& , const char*, std::string ) ;





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


/**
NP::SetMeta
-----------

Updates the single *mt* string

**/

template<typename T> inline void NP::SetMeta( std::string& mt, const char* key, T value ) // static
{
    std::stringstream nn;  // stringstream for creating the updated mt string

    std::stringstream ss;  // stringstream for parsing the initial mt string
    ss.str(mt);

    std::string s;
    char delim = ':' ;
    bool changed = false ;
    while (std::getline(ss, s))
    {
       size_t pos = s.find(delim);
       if( pos != std::string::npos )  // lines has the delim, so extract (k,v)
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
               nn << s << std::endl ;    // leaving line asis
           }
       }
       else
       {
           nn << s << std::endl ;     // leaving line as is
       }
    }
    if(!changed) nn << key << delim << value << std::endl ;  // didnt find the key, so add it
    mt = nn.str() ;
}

template void     NP::SetMeta<uint64_t>(    std::string&, const char*, uint64_t );
template void     NP::SetMeta<int>(         std::string&, const char*, int );
template void     NP::SetMeta<unsigned>(    std::string&, const char*, unsigned );
template void     NP::SetMeta<float>(       std::string&, const char*, float );
template void     NP::SetMeta<double>(      std::string&, const char*, double );
template void     NP::SetMeta<std::string>( std::string&, const char*, std::string );




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

template void     NP::set_meta<uint64_t>(const char*, uint64_t );
template void     NP::set_meta<int>(const char*, int );
template void     NP::set_meta<unsigned>(const char*, unsigned );
template void     NP::set_meta<float>(const char*, float );
template void     NP::set_meta<double>(const char*, double );
template void     NP::set_meta<std::string>(const char*, std::string );



template<typename T> inline void NP::set_meta_kv(const std::vector<std::pair<std::string, T>>& kvs )
{
    SetMetaKV(meta, kvs );
}
template<typename T> inline void NP::SetMetaKV( std::string& meta, const std::vector<std::pair<std::string, T>>& kvs ) // static
{
    for(int i=0 ; i < int(kvs.size()); i++) SetMeta(meta, kvs[i].first.c_str(), kvs[i].second );
}


template<typename T> inline std::string NP::DescKV( const std::vector<std::pair<std::string, T>>& kvs ) // static
{
    typedef std::pair<std::string, T> KV ;
    std::stringstream ss ;
    ss << "NP::DescKV" << std::endl ;
    for(INT i=0 ; i < INT(kvs.size()) ; i++)
    {
        const KV& kv = kvs[i] ;
        ss
            << std::setw(20)  << kv.first
            << " : "
            << std::setw(100) << kv.second
            << std::endl
            ;
    }
    std::string str = ss.str();
    return str ;
}


inline void NP::SetMetaKV_(
    std::string& meta,
    const std::vector<std::string>& keys,
    const std::vector<std::string>& vals ) // static
{
    assert( keys.size() == vals.size() );
    for(INT i=0 ; i < INT(keys.size()); i++) SetMeta(meta, keys[i].c_str(), vals[i].c_str() );
}

inline void NP::setMetaKV_( const std::vector<std::string>& keys,  const std::vector<std::string>& vals )
{
    SetMetaKV_(meta, keys, vals);
}







inline std::string NP::descMeta() const
{
    std::stringstream ss ;
    ss << "NP::descMeta"
       << std::endl
       << meta
       << std::endl
       ;
    std::string str = ss.str();
    return str ;
}


/**
NP::GetFirstStampIndex_OLD
---------------------------

Return index of the first stamp that has difference to
the next stamp of less than the discount. This is
to avoid uninteresting large time ranges in the deltas.

HMM: this assumes the stamps are ascending

HMM: simpler to just disqualify stamps during initialization

**/

inline NP::INT NP::GetFirstStampIndex_OLD(const std::vector<int64_t>& stamps, int64_t discount ) // static
{
    INT first = -1 ;
    INT i_prev = -1 ;
    int64_t t_prev = -1 ;

    for(INT i=0 ; i < INT(stamps.size()) ; i++)
    {
        if(stamps[i] == 0) continue ;

        int64_t t  = stamps[i] ;
        int64_t dt = t_prev > -1 ? t - t_prev : -1 ;
        if( dt > -1 && dt < discount && first == -1 ) first = i_prev ;

        t_prev = t ;
        i_prev = i ;
    }
    return first ;
}









/**
NP::MakeMetaKVS_ranges
------------------------

Former NP::makeMetaKVS_ranges turned static with meta arg.

**/

inline NP* NP::MakeMetaKVS_ranges(const std::string& meta_, const char* ranges_ , std::ostream* ss) // static
{
    std::vector<std::string> keys ;
    std::vector<std::string> vals ;
    std::vector<int64_t> tt ;
    bool only_with_stamp = true ;
    U::GetMetaKVS(meta_, &keys, &vals, &tt, only_with_stamp );
    assert( keys.size() == vals.size() );
    assert( keys.size() == tt.size() );
    assert( tt.size() == keys.size() );
    return MakeMetaKVS_ranges(keys, tt, ranges_ , ss );
}

inline NP* NP::MakeMetaKVS_ranges2(const std::string& meta_, const char* ranges_ , std::ostream* ss) // static
{
    std::vector<std::string> keys ;
    std::vector<std::string> vals ;
    std::vector<int64_t> tt ;
    bool only_with_stamp = true ;
    U::GetMetaKVS(meta_, &keys, &vals, &tt, only_with_stamp );
    assert( keys.size() == vals.size() );
    assert( keys.size() == tt.size() );
    assert( tt.size() == keys.size() );
    return MakeMetaKVS_ranges2(keys, tt, ranges_ , ss );
}





/**
NP::Resolve_ranges
-------------------

Ranges are newline delimited with colon separated pairs of tags and mandatory annotation, eg::

   SEvt__Init_RUN_META:CSGFoundry__Load_HEAD                     ## init
   CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL                   ## load_geom
   CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL                   ## upload_geom
   A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_PREL         ## upload_genstep
   A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate
   A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_TAIL         ## download

Stamp keys are wildcarded by including strings like %0.3d
so need to pre-pass looking for keys with a range of indices,
so effectively are generating simple ranges without wildcard
based on the keys, wildcards and idx range.

1. split the colon delimited range pair from the ## delimited annotation
2. generate specs vector of wildcard resolved key ranges including the annotation

The output specs are of form::

    A000_QSim__simulate_HEAD:A000_QSim__simulate_PREL:upload_genstep

**/

inline void NP::Resolve_ranges( std::vector<std::string>& specs, const std::vector<std::string>& keys, const char* ranges_, std::ostream* ss )
{
    assert(ranges_ && strlen(ranges_) > 0);
    std::vector<std::string> ranges ;
    std::vector<std::string> anno ;
    U::LiteralAnno(ranges, anno, ranges_ , "#" );
    assert( ranges.size() == anno.size() ) ;

    int num_keys = keys.size() ;
    int num_ranges = ranges.size() ;

    if(ss) ss->imbue(std::locale("")) ; // commas for thousands

    if(ss) (*ss)
       << "[NP::Resolve_ranges\n"
       << " num_keys :" << num_keys
       << " num_ranges :" << num_ranges
       << std::endl
       ;

    // generate specs of wildcard resolved ranges

    char delim = ':' ;

    for(int i=0 ; i < num_ranges ; i++)
    {
        const std::string& range = ranges[i] ;  //
        size_t pos = range.find(delim);
        if( pos == std::string::npos ) continue ;

        std::string _a = range.substr(0, pos);
        std::string _b = range.substr(pos+1);
        const char* a = _a.c_str();
        const char* b = _b.c_str();

        // idx0 idx1 specifies the range for wildcard replacements
        int idx0 = 0 ;
        int idx1 = 30 ;
        for(int idx=idx0 ; idx < idx1 ; idx++)
        {
            std::string akey ;
            std::string bkey ;
            int ia = U::FormattedKeyIndex(akey, keys, a, idx, idx+1 ) ;
            int ib = U::FormattedKeyIndex(bkey, keys, b, idx, idx+1 ) ;

            bool found_range_pair = !akey.empty() && !bkey.empty() && ia > -1 && ib > -1 ;

            if(found_range_pair)
            {
                std::stringstream mm ;
                mm << akey << ":" << bkey << ":" << anno[i] ;
                std::string spec = mm.str();
                if(std::find(specs.begin(), specs.end(), spec) == specs.end())  specs.push_back(spec);
            }
        }
    }
    int num_specs = specs.size();

    if(ss) (*ss)
       << "]NP::Resolve_ranges\n"
       << " num_keys :" << num_keys
       << " num_ranges :" << num_ranges
       << " num_specs :"  << num_specs
       << std::endl
       ;

}


/**
NP::TimeOrder_ranges
---------------------

1. collect lhs (start) times of the ranges into stt
2. sort spec_order indices into ascending time order

**/


inline void NP::TimeOrder_ranges( std::vector<int>& spec_order, const std::vector<std::string>& specs, const std::vector<std::string>& keys, const std::vector<int64_t>& tt, std::ostream* ss )
{
    assert( spec_order.size() == specs.size() ) ;
    int num_specs = specs.size();

    std::vector<int64_t> stt(num_specs);

    for(int i=0 ; i < num_specs ; i++)
    {
        const char* spec = specs[i].c_str();
        std::vector<std::string> elem ;
        U::Split( spec, ':', elem );
        assert( elem.size() > 1 );

        const char* ak = elem[0].c_str();
        const char* bk = elem[1].c_str();

        int ia = U::KeyIndex( keys, ak );
        int ib = U::KeyIndex( keys, bk );

        int64_t ta = ia > -1 ? tt[ia] : 0 ;
        int64_t tb = ib > -1 ? tt[ib] : 0 ;

        bool expect = ta > 0 && tb > 0 ;
        if(!expect) std::cerr << "NP::TimeOrder_ranges MISSING KEY " << std::endl ;
        assert(expect );

        stt[i] = ta ;
    }

    // Sort indices into ascending start time order

    std::iota(spec_order.begin(), spec_order.end(), 0);
    auto order = [&stt](const size_t& a, const size_t &b) { return stt[a] < stt[b];}  ;
    std::sort(spec_order.begin(), spec_order.end(), order );
}

inline NP* NP::MakeMetaKVS_ranges_table(
    const std::vector<int>& spec_order,
    const std::vector<std::string>& specs,
    const std::vector<std::string>& keys,
    const std::vector<int64_t>& tt,
    std::ostream* ss )
{
    assert( spec_order.size() == specs.size() ) ;
    int num_specs = specs.size();

    int ni = num_specs ;
    int nj = 5 ;

    NP* _rr = NP::Make<int64_t>( ni, nj ) ;
    int64_t* rr = _rr->values<int64_t>();

    int64_t ab_total = 0 ;
    int wid = 30 ;
    _rr->labels = new std::vector<std::string> { "ta", "tb", "ab", "ia", "ib" } ;

    for(int j=0 ; j < num_specs ; j++)
    {
        int i = spec_order[j];
        const char* spec = specs[i].c_str();
        _rr->names.push_back(spec);

        std::vector<std::string> elem ;
        U::Split( spec, ':', elem );
        assert( elem.size() > 1 );

        const char* ak = elem[0].c_str();
        const char* bk = elem[1].c_str();
        const char* no = elem.size() > 2 ? elem[2].c_str() : nullptr ;

        int ia = U::KeyIndex( keys, ak );
        int ib = U::KeyIndex( keys, bk );

        int64_t ta = ia > -1 ? tt[ia] : 0 ;
        int64_t tb = ib > -1 ? tt[ib] : 0 ;
        int64_t ab = tb - ta ;

        rr[nj*j+0] = ta ;
        rr[nj*j+1] = tb ;
        rr[nj*j+2] = ab ;
        rr[nj*j+3] = ia ;
        rr[nj*j+4] = ib ;

        ab_total += ab ;

        if(ss) (*ss)
            << " " << std::setw(wid) << ak
            << " ==> "
            << " " << std::setw(wid) << bk
            << "      " << std::setw(16) << std::right << ab
            << "      " << std::setw(16) << std::right << ta
            << "      " << std::setw(16) << std::right << tb
            << ( no == nullptr ? "" : "    ## " ) << ( no ? no : "" )
            << std::endl
            ;
    }

    if(ss) (*ss)
       << " " << std::setw(wid) << ""
       << "     "
       << " " << std::setw(wid) << "TOTAL:"
       << "      " << std::setw(16) << std::right << ab_total
       << std::endl
       ;

    return _rr ;
}

/**
NP::MakeMetaKVS_ranges2_table
------------------------------

1. For each range lookup the a and b key indices into keys vector.
2. When the number of indices for a and b matches simply collect indices and time stamps for the range into the kpp tuple vector
3. When one indice is found for b and more than one for a find the a index closest in absolute time difference to b
4. Ditto for a and b situation reversed
5. sort indices of the kpp tuple into ascending a(start) time order

**/

inline NP* NP::MakeMetaKVS_ranges2_table( const std::vector<std::string>& specs, const std::vector<std::string>& keys, const std::vector<int64_t>& tt, std::ostream* ss )
{
    int num_specs = specs.size();
    if(ss) (*ss) << "[NP::MakeMetaKVS_ranges2_table num_specs " << num_specs << "\n";

    using KP = std::tuple<int,int,int64_t,int64_t,const char*, int> ;

    std::vector<KP> kpp ;

    for(int i=0 ; i < num_specs ; i++)
    {
        const char* spec = specs[i].c_str();
        std::vector<std::string> elem ;
        U::Split( spec, ':', elem );
        assert( elem.size() > 1 );

        const char* ak = elem[0].c_str();
        const char* bk = elem[1].c_str();
        const char* no = elem.size() > 2 ? elem[2].c_str() : nullptr ;
        const char* uno = no ? strdup(no) : nullptr ;

        std::vector<int> iia ;
        U::KeyIndices(iia, keys, ak );

        std::vector<int> iib ;
        U::KeyIndices(iib, keys, bk );

        int na = iia.size();
        int nb = iib.size();

        if(na == nb)
        {
            for(int j=0 ; j < na ; j++) kpp.push_back({iia[j],iib[j],tt[iia[j]],tt[iib[j]], uno, i });
        }
        else if( na > 1 && nb == 1 )
        {
            // pick smallest absolute delta between the multiple a and single b
            std::vector<int64_t> att ;
            for(int a=0 ; a < na ; a++) att.push_back( std::abs( tt[iia[a]] - tt[iib[0]] ) );
            int ia = std::distance(std::begin(att), std::min_element(std::begin(att), std::end(att)));
            kpp.push_back( {iia[ia], iib[0], tt[iia[ia]], tt[iib[0]], uno, i } );
        }
        else if( nb > 1 && na == 1 )
        {
            // pick smallest absolute delta between the multiple b and single a
            std::vector<int64_t> btt ;
            for(int b=0 ; b < nb ; b++) btt.push_back( std::abs( tt[iib[b]] - tt[iia[0]] ) );
            int ib = std::distance(std::begin(btt), std::min_element(std::begin(btt), std::end(btt)));
            kpp.push_back( {iia[0], iib[ib], tt[iia[0]], tt[iib[ib]], uno, i } );
        }
    }

    int num_kpp = kpp.size();
    if(ss) (*ss) << ".NP::MakeMetaKVS_ranges2_table kpp.size " << kpp.size() << "\n" ;

    // Sort indices into ascending start time order
    std::vector<int> ikp(num_kpp);
    std::iota(ikp.begin(), ikp.end(), 0);
    auto kp_order = [&kpp](const size_t& a, const size_t &b) { return std::get<2>(kpp[a]) < std::get<2>(kpp[b]);}  ;
    std::sort(ikp.begin(), ikp.end(), kp_order );


    int dbg = 0 ;
    int ni = num_kpp ;
    int nj = 5 ;

    NP* _rr = NP::Make<int64_t>( ni, nj ) ;
    int64_t* rr = _rr->values<int64_t>();

    int wid = 30 ;
    _rr->labels = new std::vector<std::string> { "ta", "tb", "ab", "ia", "ib" } ;

    int64_t ab_total = 0 ;

    std::vector<std::tuple<int,int64_t>> abs ;

    for(int j=0 ; j < num_kpp ; j++)
    {
        int i = ikp[j] ;

        const KP& kp = kpp[i];
        int ia = std::get<0>(kp) ;
        int ib = std::get<1>(kp) ;
        int64_t ta = std::get<2>(kp) ;
        int64_t tb = std::get<3>(kp) ;
        const char* no = std::get<4>(kp) ;
        int ispec = std::get<5>(kp) ;    // when ispec stays the same its a repeated range
        const char* spec = specs[ispec].c_str();
        _rr->names.push_back(spec);

        int64_t ab = tb - ta ;
        abs.push_back( {ispec, ab } );

        int64_t ab_cumsum = 0 ;  // sum ab so far with the same spec
        int ab_cumsum_num = 0 ;
        for(int i=0 ; i < int(abs.size()) ; i++)
        {
            if( std::get<0>(abs[i]) == ispec )
            {
                ab_cumsum += std::get<1>(abs[i]) ;
                ab_cumsum_num += 1 ;
            }
        }
        bool rep = ab_cumsum_num > 1 ;


        ab_total += ab ;

        rr[nj*j+0] = ta ;
        rr[nj*j+1] = tb ;
        rr[nj*j+2] = ab ;
        rr[nj*j+3] = ia ;
        rr[nj*j+4] = ib ;

        if(ss) (*ss)
            << " " << std::setw(wid) << keys[ia]
            << " ==> "
            << " " << std::setw(wid) << keys[ib]
            << "      " << std::setw(16) << std::right << ab
            ;

        if(ss && dbg > 0 ) (*ss)
            << "      " << std::setw(16) << std::right << ta
            << "      " << std::setw(16) << std::right << tb
            ;

        if(ss) (*ss)
            << ( rep ? " REP " : "     " )
            ;

        if(ss && rep) (*ss)
            << "      " << std::setw(16) << std::right << ab_cumsum
            ;

        if(ss && !rep) (*ss)
            << "      " << std::setw(16) << ""
            ;

        if(ss) (*ss)
            << ( no == nullptr ? "" : "    ## " ) << ( no ? no : "" )
            << std::endl
            ;
    }

    if(ss) (*ss)
       << " " << std::setw(wid) << ""
       << "     "
       << " " << std::setw(wid) << "TOTAL:"
       << "      " << std::setw(16) << std::right << ab_total
       << std::endl
       ;

    if(ss) (*ss)
       << "]NP::MakeMetaKVS_ranges2_table num_keys:" << keys.size() << "\n"
       ;

    return _rr ;
}









/**
NP::MakeMetaKVS_ranges
-----------------------

Does not handle repeated keys well, as happens with multi-launch.
See ranges2 which attempts to fix that.

**/

inline NP* NP::MakeMetaKVS_ranges( const std::vector<std::string>& keys, const std::vector<int64_t>& tt, const char* ranges_, std::ostream* ss )
{
    std::vector<std::string> specs ;
    Resolve_ranges(specs, keys, ranges_, ss );
    int num_specs = specs.size();

    if(ss) (*ss)
       << ".NP::MakeMetaKVS_ranges num_specs:" << num_specs << "\n"
       ;

    std::vector<int> spec_order(num_specs) ;
    TimeOrder_ranges( spec_order, specs, keys, tt, ss );

    // present the ranges in order of start time
    NP* rr = MakeMetaKVS_ranges_table(spec_order, specs, keys, tt, ss );

    return rr ;
}


/**
NP::MakeMetaKVS_ranges2
-------------------------

HMM: with repeated keys which currently happens with multi-launch
this code is unaware of that and just acts on the timestamp of
the first key found. It would be more meaningful to find all the ranges
and sum them, with reporting on how many were summed.

**/

inline NP* NP::MakeMetaKVS_ranges2( const std::vector<std::string>& keys, const std::vector<int64_t>& tt, const char* ranges_, std::ostream* ss )
{
    if(ss) (*ss)
       << "[NP::MakeMetaKVS_ranges2 num_keys:" << keys.size() << "\n"
       ;

    std::vector<std::string> specs ;
    Resolve_ranges(specs, keys, ranges_, ss );
    int num_specs = specs.size();

    NP* rr = MakeMetaKVS_ranges2_table(specs, keys, tt, ss );

    if(ss) (*ss)
       << "]NP::MakeMetaKVS_ranges2 num_specs:" << num_specs << " rr " << ( rr ? rr->sstr() : "-" ) << "\n"
       ;

    return rr ;
}







inline std::string NP::DescMetaKVS_kvs(const std::vector<std::string>& keys, const std::vector<std::string>& vals, const std::vector<int64_t>& tt )  // static
{
    int num_keys = keys.size() ;
    if(num_keys == 0) return "" ;

    // sort indices into increasing time order
    // non-timestamped lines with placeholder timestamp zero will come first
    std::vector<int> ii(num_keys);
    std::iota(ii.begin(), ii.end(), 0);  // init to 0,1,2,3,..., num_keys-1
    auto order = [&tt](const size_t& a, const size_t &b) { return tt[a] < tt[b];}  ;
    std::sort(ii.begin(), ii.end(), order );

    std::stringstream ss ;
    ss.imbue(std::locale("")) ;  // commas for thousands

    int64_t t_first = 0 ;
    int64_t t_second = 0 ;
    int64_t t_prev  = 0 ;

    int t_count = 0 ;

    ss
        << "[NP::DescMetaKVS_kvs "
        << " keys.size " << keys.size()
        << " vals.size " << vals.size()
        << " tt.size "   << tt.size()
        << " num_keys " << num_keys
        << "\n"
        ;

    for(int j=0 ; j < num_keys ; j++)
    {
        int i = ii[j] ;
        const char* k = keys[i].c_str();
        const char* v = vals[i].c_str();
        int64_t     t = tt[i] ;

        if(t_first > 0 && t_second == 0 && t > 0 ) t_second = t  ;
        if(t_first == 0 && t > 0 ) t_first = t  ;

        int64_t dt0 = t > 0 && t_first  > 0 ? t - t_first  : -1 ; // microseconds since first
        int64_t dt1 = t > 0 && t_second > 0 ? t - t_second : -1 ; // microseconds since second
        int64_t dt  = t > 0 && t_prev   > 0 ? t - t_prev   : -1 ; // microseconds since previous stamp
        if(t > 0) t_prev = t ;
        if(t > 0) t_count += 1 ;
        if(t_count == 1 ) ss
            << "\n"
            << std::setw(30) << "k"
            << " : "
            << std::setw(35) << "v"
            << "   "
            << std::setw(27) << (  "t:microsecond"  )
            << " " << std::setw(11) << "dt0:(t-t0)"
            << " " << std::setw(11) << "dt1:(t-t1)"
            << " " << std::setw(11) << "dt:(t-tpr)"
            << std::endl
            ;


        ss << std::setw(30) << k
           << " : "
           << std::setw(35) << v
           << "   "
           << std::setw(27) << (  t > 0 ? U::Format(t) : "" )
           << " " << std::setw(11) << U::FormatInt(dt0, 11)
           << " " << std::setw(11) << U::FormatInt(dt1, 11)
           << " " << std::setw(11) << U::FormatInt(dt , 11 )
           << std::endl
           ;
    }
    ss << "]NP::DescMetaKVS_kvs\n" ;
    std::string str = ss.str();
    return str ;
}


/**
NP::DescMetaKVS_juncture
-------------------------

Example juncture::

    SEvt__Init_RUN_META,SEvt__BeginOfRun,SEvt__EndOfRun,SEvt__Init_RUN_META

1. split juncture string on comma delimiters
2. loop over juncture j_key looking up the corresponding KeyIndex
3. report deltas between the timestamps looked up from the juncture keys

Essentially this is just a selected key version of the full descMetaKVS
listing that precedes it which can be easier to understand with careful
choice of juncture keys appropriate for the parts of the code that
take the time. The ordering is entirely from the input juncture key
order with no sorting. So delta times can be negative.

**/


inline std::string NP::DescMetaKVS_juncture( const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* juncture_ )
{
    assert(juncture_ && strlen(juncture_) > 0);

    int it_first = std::distance(std::begin(tt), std::min_element(std::begin(tt), std::end(tt)));
    int64_t t0 = tt[it_first];

    std::vector<std::string> juncture ;
    Split(juncture, juncture_ , ',' );
    INT num_juncture = juncture.size() ;

    std::stringstream ss ;
    ss << "[NP::DescMetaKVS_juncture\n" ;
    ss << "num_juncture " << num_juncture << "\n" ;
    ss << "juncture [" << juncture_ << "] time ranges between junctures" << std::endl ;
    ss.imbue(std::locale("")) ;  // commas for thousands


    ss << std::setw(30) << "k"
       << " : "
       << std::setw(12) << "dtp"
       << std::setw(23) << ""
       << " : "
       << std::setw(12) << "dt0"
       << " : "
       << "timestamp"
       << std::endl
       ;

    int64_t tp = 0 ;
    for(int j=0 ; j < num_juncture ; j++)
    {
        const char* j_key = juncture[j].c_str() ;
        int i = U::KeyIndex(keys, j_key) ;
        if( i == -1 ) continue ;

        const char* k = keys[i].c_str();
        int64_t t = tt[i] ;

        int64_t dtp = ( t > 0 && tp > 0 ) ? t - tp : -1  ;
        int64_t dt0 = ( t > 0 && t0 > 0 ) ? t - t0 : -1 ;

        ss << std::setw(30) << k
           << " : "
           << std::setw(12) << dtp
           << std::setw(23) << ""
           << " : "
           << std::setw(12) << dt0
           << " : "
           << U::Format(t)
           << " JUNCTURE"
           << std::endl
           ;

         if( t > 0 ) tp = t ;
    }
    ss << "]NP::DescMetaKVS_juncture\n" ;
    std::string str = ss.str();
    return str ;
}

/**
NP::DescMetaKVS_ranges
------------------------

This does not handle repeated keys, eg from multi-launch running.

**/

inline std::string NP::DescMetaKVS_ranges( const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* ranges_ )
{
    std::stringstream ss ;
    ss << "[NP::DescMetaKVS_ranges\n"
       << "[ranges\n"
       << ( ranges_ ? ranges_ : "-" )
       << "]ranges\n"
       ;

    NP* a = MakeMetaKVS_ranges(keys, tt, ranges_ , &ss );

    ss << "]NP::DescMetaKVS_ranges"
       << " a " << ( a ? a->sstr() : "-" )
       << "\n"
       ;

    std::string str = ss.str();
    return str ;
}


/**
NP::DescMetaKVS_ranges2
------------------------

This attempts to handle repeated keys reasonably, eg from multi-launch running.

**/

inline std::string NP::DescMetaKVS_ranges2( const std::vector<std::string>& keys, std::vector<int64_t>& tt, const char* ranges_ )
{
    std::stringstream ss ;
    ss << "[NP::DescMetaKVS_ranges2\n"
       << "[ranges\n"
       << ( ranges_ ? ranges_ : "-" )
       << "]ranges\n"
       ;

    NP* a = MakeMetaKVS_ranges2(keys, tt, ranges_ , &ss );

    ss << "]NP::DescMetaKVS_ranges2"
       << " a " << ( a ? a->sstr() : "-" )
       << "\n"
       ;

    std::string str = ss.str();
    return str ;
}





/**
NP::DescMetaKVS
----------------

1. GetMetaKVS extracting key, val pairs and microsecond timestamps,
   lines without 16 digit timestamps have placeholder timestamps of zero
2. std::iota create vector of indices 0,1,2,3...num_key - 1
3. sort indices into increasing timestamp order, all placeholder zeros will be at start
4. report time stamps with deltas from 1st, 2nd and previous
5. DescMetaKVS_juncture
6. DescMetaKVS_ranges

**/


inline std::string NP::DescMetaKVS(const std::string& meta, const char* juncture_ , const char* ranges_ )  // static
{
    VS keys ;
    VS vals ;
    VT tt ;

    bool only_with_stamp = false ;

    U::GetMetaKVS(meta, &keys, &vals, &tt, only_with_stamp );

    assert( keys.size() == vals.size() );
    assert( keys.size() == tt.size() );
    assert( tt.size() == keys.size() );

    std::stringstream ss ;
    ss << "[NP::DescMetaKVS only_with_stamp : " << ( only_with_stamp ? "YES" : "NO " ) << "\n" ;

    ss << DescMetaKVS_kvs( keys, vals, tt ) ;
    if(juncture_ && strlen(juncture_) > 0 ) ss << DescMetaKVS_juncture(keys, tt, juncture_ );
    if(ranges_ && strlen(ranges_) > 0 )     ss << DescMetaKVS_ranges2(keys, tt, ranges_ );

    ss << "]NP::DescMetaKVS\n" ;

    std::string str = ss.str();
    return str ;
}


inline std::string NP::descMetaKVS(const char* juncture_, const char* ranges_) const
{

    bool dump_meta = false ;

    std::stringstream ss ;
    ss << "[NP::descMetaKVS\n" ;

    if(dump_meta) ss
       << "[meta\n"
       << meta
       << "]meta\n"
       ;

    ss
       << DescMetaKVS(meta, juncture_, ranges_)
       << "]NP::descMetaKVS\n"
       ;
    std::string str = ss.str();
    return str ;
}








inline std::string NP::DescMetaKV(const std::string& meta, const char* juncture_, const char* ranges_ )  // static
{
    std::vector<std::string> keys ;
    std::vector<std::string> vals ;
    bool only_with_profile = false ;
    GetMetaKV(meta, &keys, &vals, only_with_profile );
    assert( keys.size() == vals.size() );
    INT num_keys = keys.size();

    int64_t t0 = std::numeric_limits<int64_t>::max() ;
    std::vector<int64_t> tt ;
    std::vector<INT> ii ;

    // collect times and indices of all entries
    // time is set to zero for entries without time stamps
    for(INT i=0 ; i < num_keys ; i++)
    {
        const char* v = vals[i].c_str();
        bool looks_like_stamp = U::LooksLikeStampInt(v);
        bool looks_like_prof  = U::LooksLikeProfileTriplet(v);
        int64_t t = 0 ;
        if(looks_like_stamp) t = U::To<int64_t>(v) ;
        if(looks_like_prof)  t = strtoll(v, nullptr, 10);
        tt.push_back(t);
        ii.push_back(i);
        if(t > 0 && t < t0) t0 = t ;
    }

    // sort the indices into time increasing order
    auto order = [&tt](const size_t& a, const size_t &b) { return tt[a] < tt[b];}  ;
    std::sort( ii.begin(), ii.end(), order );


    std::stringstream ss ;
    ss.imbue(std::locale("")) ;  // commas for thousands

    // use the time sorted indices to output in time order
    // entries without time info at t=0 appear first
    for(INT j=0 ; j < num_keys ; j++)
    {
        INT i = ii[j] ;
        const char* k = keys[i].c_str();
        const char* v = vals[i].c_str();
        int64_t t = tt[i] ;

        ss << std::setw(30) << k
           << " : "
           << std::setw(35) << v
           << " : "
           << std::setw(12) << ( t > 0 ? t - t0 : -1 )
           << " : "
           << ( t > 0 ? U::Format(t) : "" )
           << std::endl
           ;
    }



    if(juncture_ && strlen(juncture_) > 0)
    {
        std::vector<std::string> juncture ;
        Split(juncture, juncture_ , ',' );
        INT num_juncture = juncture.size() ;
        ss << "juncture:" << num_juncture << " [" << juncture_ << "] time ranges between junctures" << std::endl ;

        int64_t tp = 0 ;
        for(INT j=0 ; j < num_juncture ; j++)
        {
            const char* j_key = juncture[j].c_str() ;
            INT i = std::distance( keys.begin(), std::find(keys.begin(), keys.end(), j_key )) ;
            if( i == INT(keys.size()) ) continue ;

            const char* k = keys[i].c_str();
            //const char* v = vals[i].c_str();
            int64_t t = tt[i] ;

            ss << std::setw(30) << k
               << " : "
               << std::setw(12) << ( t > 0 && tp > 0 ? t - tp : -1 )
               << std::setw(23) << ""
               << " : "
               << std::setw(12) << ( t > 0 && t0 > 0 ? t - t0 : -1 )
               << " : "
               << U::Format(t)
               << " JUNCTURE"
               << std::endl
               ;

             if( t > 0 ) tp = t ;
        }
    }

    std::string str = ss.str();
    return str ;
}

inline std::string NP::descMetaKV(const char* juncture, const char* ranges) const
{
    std::stringstream ss ;
    ss << "NP::descMetaKV"
       << std::endl
       << DescMetaKV(meta, juncture, ranges)
       ;
    std::string str = ss.str();
    return str ;
}



inline const char* NP::get_lpath() const
{
    return lpath.c_str() ? lpath.c_str() : "-" ;
}


template<typename T>
inline NP::INT NP::DumpCompare( const NP* a, const NP* b , unsigned a_column, unsigned b_column, const T epsilon ) // static
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
    INT mismatch = 0 ;

    for(unsigned i=0 ; i < a_ni ; i++)
    {
        const T av = aa[a_nj*i+a_column] ;
        const T bv = bb[b_nj*i+b_column] ;
        av_sum += av ;
        bv_sum += bv ;

        bool is_diff = std::abs(av-bv) > epsilon ;
        if(is_diff) std::cout
            << std::setw(4) << i
            << " a " << std::setw(10) << std::fixed << std::setprecision(4) << av
            << " b " << std::setw(10) << std::fixed << std::setprecision(4) << bv
            << " a-b " << std::setw(10) << std::fixed << std::setprecision(4) << av-bv
            << std::endl
            ;
        if(is_diff) mismatch += 1 ;
    }
    if(mismatch > 0) std::cout
        << "NP::DumpCompare "
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

inline NP::INT NP::Memcmp(const NP* a, const NP* b ) // static
{
    unsigned a_bytes = a ? a->arr_bytes() : 0 ;
    unsigned b_bytes = b ? b->arr_bytes() : 0 ;
    return a_bytes == b_bytes && a_bytes > 0 ? memcmp(a->bytes(), b->bytes(), a_bytes) : -1 ;
}

inline bool NP::SameData( const NP* a, const NP* b )
{
    return 0 == Memcmp(a, b );
}

/**
NP::Concatenate
----------------

Load the named NP arrays from directory and concatenate them in name order

**/


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

/**
NP::Concatenate
----------------

* template allows same code to work with both "NP" and "const NP"
* arrays must have the same number of itemvalues, ie values after first dimension


**/

template<typename T>
inline NP* NP::Concatenate(const std::vector<T*>& aa )  // static
{
    [[maybe_unused]] INT num_a = aa.size();
    assert( num_a > 0 );
    auto a0 = aa[0] ;

    unsigned nv0 = a0->num_itemvalues() ;
    const char* dtype0 = a0->dtype ;

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        auto a = aa[i] ;

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

    std::vector<INT> comb_shape ;
    NPS::copy_shape( comb_shape, a0->shape );
    comb_shape[0] = ni_total ;

    NP* c = new NP(a0->dtype);
    c->set_shape(comb_shape);
    if(VERBOSE) std::cout << "NP::Concatenate c " << c->desc() << std::endl ;

    unsigned offset_bytes = 0 ;
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        auto a = aa[i];
        unsigned a_bytes = a->arr_bytes() ;
        memcpy( c->data.data() + offset_bytes ,  a->data.data(),  a_bytes );
        offset_bytes += a_bytes ;
        //a->clear(); // HUH: THAT WAS IMPOLITE : ASSUMING CALLER DOESNT WANT TO USE INPUTS
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


annotate:true && parasite!=nullptr
    parasite array:

    * must be 1d
    * same length as aa vector : ie one value is provided per input array
    * same dtype as the arrays

    The single parasitic values per input array are incorporated
    into the -2 item slot in the combined array.

**/
inline NP* NP::Combine(const std::vector<const NP*>& aa, bool annotate, const NP* parasite)  // static
{
    assert( aa.size() > 0 );
    const NP* a0 = aa[0] ;

    const char* dtype0 = a0->dtype ;
    INT ebyte0 = a0->ebyte ;
    unsigned ndim0 = a0->shape.size() ;
    unsigned ldim0 = a0->shape[ndim0-1] ;
    unsigned fdim_mx = a0->shape[0] ;

    for(unsigned i=1 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i];
        bool dtype_expect = strcmp( a->dtype, dtype0 ) == 0  ;
        if(!dtype_expect) std::cerr << "NP::Combine : input arrays must all have same dtype " << std::endl;
        assert( dtype_expect );

        unsigned ndim = a->shape.size() ;
        bool ndim_expect = ndim == ndim0  ;
        if(!ndim_expect) std::cerr << "NP::Combine : input arrays must all have an equal number of dimensions " << std::endl;
        assert( ndim_expect );

        unsigned ldim = a->shape[ndim-1] ;
        bool ldim_expect = ldim == ldim0 ;
        if(!ldim_expect) std::cerr << "NP::Combine : last dimension of the input arrays must be equal " << std::endl ;
        assert( ldim_expect );

        unsigned fdim = a->shape[0] ;
        if( fdim > fdim_mx ) fdim_mx = fdim ;
    }


    if(parasite)
    {
        assert( parasite->shape.size() == 1 && parasite->shape[0] == INT(aa.size()) );
        assert( strcmp( parasite->dtype, dtype0) == 0 && "parasite arrays must have same dtype as those being combined" );
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
    unsigned item_values = item_bytes/ebyte0 ;

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
            const float* pp = parasite ? parasite->cvalues<float>() : nullptr ;

            UIF32 uif32 ;
            for(unsigned i=0 ; i < aa.size() ; i++)
            {
                const NP* a = aa[i];
                uif32.u = a->shape[0] ;
                if(VERBOSE) std::cout << "NP::Combine annotate " << i << " uif32.u  " << uif32.u  << std::endl ;
                *(cc + (i+1)*item_values - 1) = uif32.f ;
                if(pp) *(cc + (i+1)*item_values - 2) = pp[i] ;
                // (i+1)*item_bytes/ebyte0 is off the edge, then -1 to be the last value
            }
        }
        else if( ebyte0 == 8 )
        {
            double* cc = c->values<double>();
            const double* pp = parasite ? parasite->cvalues<double>() : nullptr ;

            UIF64 uif64 ;
            for(unsigned i=0 ; i < aa.size() ; i++)
            {
                const NP* a = aa[i];
                uif64.u = a->shape[0] ;
                if(VERBOSE) std::cout << "NP::Combine annotate " << i << " uif64.u  " << uif64.u  << std::endl ;
                *(cc + (i+1)*item_values - 1) = uif64.f ;
                if(pp) *(cc + (i+1)*item_values - 2) = pp[i] ;
            }

            c->set_preserve_last_column_integer_annotation() ;
            // make the annotation survive MakeNarrow
            // (currently annotation is scrubbed by MakeWide but could be easily be implented)
        }
    }
    return c ;
}

template<typename... Args> inline NP* NP::Combine_(Args ... args)  // Combine_ellipsis
{
    std::vector<const NP*> aa = {args...};
    bool annotate = true ;
    return Combine(aa, annotate);
}


inline bool NP::Exists(const char* base, const char* rel,  const char* name) // static
{
    std::string path = U::form_path(base, rel, name);
    return Exists(path.c_str());
}
inline bool NP::Exists(const char* dir, const char* name) // static
{
    std::string path = U::form_path(dir, name);
    return Exists(path.c_str());
}
inline bool NP::Exists(const char* path_) // static
{
    const char* path = U::Resolve(path_);
    std::ifstream fp(path, std::ios::in|std::ios::binary);
    return fp.fail() ? false : true ;
}

inline bool NP::ExistsSidecar( const char* path, const char* ext ) // static
{
    std::string vstr_path = U::ChangeExt(path, ".npy", ext );
    return Exists(vstr_path.c_str()) ;
}



inline bool NP::IsNoData(const char* path) // static
{
    return path && strlen(path) > 0 && path[0] == NODATA_PREFIX ;
}

inline const char* NP::PathWithNoDataPrefix(const char* path) // static
{
    if(path == nullptr) return nullptr ;
    if(IsNoData(path)) return path ;   // dont add prefix if one already present

    std::stringstream ss ;
    ss << NODATA_PREFIX << path ;
    std::string str = ss.str() ;
    return strdup(str.c_str());
}


inline int NP::load(const char* dir, const char* name)
{
    std::string path = U::form_path(dir, name);
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

inline int NP::load(const char* _path)
{
    nodata = IsNoData(_path) ;  // _path starting with NODATA_PREFIX currently '@'
    const char* path = nodata ? _path + 1 : _path ;

    if(VERBOSE) std::cerr << "[ NP::load " << path << std::endl ;

    lpath = path ;  // loadpath
    lfold = U::DirName(path);

    std::ifstream fp(path, std::ios::in|std::ios::binary);
    if(fp.fail())
    {
        std::cerr << "NP::load Failed to load from path " << path << std::endl ;
        std::raise(SIGINT);
        return 1 ;
    }

    std::getline(fp, _hdr );
    _hdr += '\n' ;

    decode_header();

    if(nodata)
    {
        if(VERBOSE) std::cerr << "NP::load SKIP reading data as nodata:true : data.size() " << data.size() << std::endl ;
    }
    else
    {
        fp.read(bytes(), arr_bytes() );
    }

    load_meta( path );
    load_names( path );
    load_labels( path );

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

inline int NP::load_strings_( const char* path, const char* ext, std::vector<std::string>* vstr )
{
    if(vstr == nullptr) return 1 ;
    std::string vstr_path = U::ChangeExt(path, ".npy", ext );
    std::ifstream fp(vstr_path.c_str(), std::ios::in);
    int rc = fp.fail() ? 1 : 0 ;

    if(false) std::cout
        << "NP::load_strings_" << std::endl
        << " path " << ( path ? path : "-" ) << std::endl
        << " vstr_path " << vstr_path << std::endl
        << " rc " << rc << std::endl
        ;

    std::string line ;
    while (std::getline(fp, line)) vstr->push_back(line);  // getline swallows new lines
    return 0 ;
}


inline int NP::load_meta(  const char* path ){  return load_string_( path, "_meta.txt",  meta  ) ; }
inline int NP::load_names( const char* path ){  return load_strings_( path, "_names.txt", &names ) ; }
inline int NP::load_labels( const char* path )
{
    labels = ExistsSidecar(path, "_labels.txt") ? new std::vector<std::string> : nullptr ;
    return load_strings_( path, "_labels.txt", labels ) ;
}


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


inline void NP::save_meta(  const char* path) const { save_string_(path, "_meta.txt",  meta  );  }
inline void NP::save_names( const char* path) const { save_strings_(path, "_names.txt", names );  }
inline void NP::save_labels(const char* path) const { if(labels) save_strings_(path, "_labels.txt", *labels );  }


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

inline void NP::save(const char* path_) const
{
    const char* path = U::Resolve(path_);  // path is nullptr with unexpanded envvar token
    if(path == nullptr) std::cerr << "NP::save failed to U::Resolve path_ " << ( path_ ? path_ : "-" ) << std::endl ;
    if(path == nullptr) return ;

    int rc = U::MakeDirsForFile(path);
    const char* _save_VERBOSE = "NP__save_VERBOSE" ;
    bool save_VERBOSE = getenv(_save_VERBOSE) != nullptr ;

    if(VERBOSE||save_VERBOSE) std::cout
          << "NP::save"
          << " " << _save_VERBOSE << ":" << ( save_VERBOSE ? "YES" : "NO " )
          << " path [" << ( path ? path : "-" ) << "]"
          << " rc:" << rc
          << "\n"
          ;

    assert( rc == 0 );

    std::string hdr = make_header();
    std::ofstream fpa(path, std::ios::out|std::ios::binary);
    fpa << hdr ;
    fpa.write( bytes(), arr_bytes() );

    save_meta( path);
    save_names(path);
    save_labels(path);
}

inline void NP::save(const char* dir, const char* reldir, const char* name) const
{
    if(VERBOSE) std::cout << "NP::save dir [" << ( dir ? dir : "-" )  << "] reldir [" << ( reldir ? reldir : "-" )  << "] name [" << name << "]" << std::endl ;
    std::string path = U::form_path(dir, reldir, name);
    save(path.c_str());
}

inline void NP::save(const char* dir, const char* name) const
{
    if(dir == nullptr || name == nullptr) std::cerr << "NP::save FAIL dir OR name arg is null " << std::endl ;
    if(dir == nullptr || name == nullptr) return ;

    std::string path = U::form_path(dir, name);
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
    std::string path = U::form_path(dir, name);
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
template <typename T> inline void NP::_dump(INT i0_, INT i1_, INT j0_, INT j1_ ) const
{
    INT ni = NPS::ni_(shape) ;  // ni_ nj_ nk_ returns shape dimension size or 1 if no such dimension
    INT nj = NPS::nj_(shape) ;
    INT nk = NPS::nk_(shape) ;

    INT i0 = i0_ == -1 ? 0                : i0_ ;
    INT i1 = i1_ == -1 ? std::min(ni, TEN) : i1_ ;

    INT j0 = j0_ == -1 ? 0                : j0_ ;
    INT j1 = j1_ == -1 ? std::min(nj, TEN) : j1_ ;


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

    for(INT i=i0 ; i < i1 ; i++){
        std::cout << "[" << std::setw(4) << i  << "] " ;
        for(INT j=j0 ; j < j1 ; j++){
            for(INT k=0 ; k < nk ; k++)
            {
                INT index = i*nj*nk + j*nk + k ;
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
    for(INT i=0 ; i < sh.ni_() ; i++ )
    for(INT j=0 ; j < sh.nj_() ; j++ )
    for(INT k=0 ; k < sh.nk_() ; k++ )
    for(INT l=0 ; l < sh.nl_() ; l++ )
    for(INT m=0 ; m < sh.nm_() ; m++ )
    for(INT o=0 ; o < sh.no_() ; o++ )
    {
        INT index = sh.idx(i,j,k,l,m,o);
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




template <typename T> void NP::Write(const char* dir, const char* reldir, const char* name, const T* data, INT ni_, INT nj_, INT nk_, INT nl_, INT nm_, INT no_ ) // static
{
    std::string path = U::form_path(dir, reldir, name);
    Write( path.c_str(), data, ni_, nj_, nk_, nl_, nm_, no_ );
}

template <typename T> void NP::Write(const char* dir, const char* name, const T* data, INT ni_, INT nj_, INT nk_, INT nl_, INT nm_, INT no_ ) // static
{
    std::string path = U::form_path(dir, name);
    Write( path.c_str(), data, ni_, nj_, nk_, nl_, nm_, no_ );
}

template <typename T> void NP::Write(const char* path, const T* data, INT ni_, INT nj_, INT nk_, INT nl_, INT nm_, INT no_ ) // static
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




template void NP::Write<float>(   const char*, const char*, const float*,        INT, INT, INT, INT, INT, INT );
template void NP::Write<double>(  const char*, const char*, const double*,       INT, INT, INT, INT, INT, INT );
template void NP::Write<int>(     const char*, const char*, const int*,          INT, INT, INT, INT, INT, INT );
template void NP::Write<unsigned>(const char*, const char*, const unsigned*,     INT, INT, INT, INT, INT, INT );


template<typename T> void NP::Write(const char* dir, const char* name, const std::vector<T>& values )
{
    if(values.size() > 0) NP::Write(dir, name, values.data(), values.size() );
}

template void NP::Write<float>(   const char*, const char*, const std::vector<float>& );
template void NP::Write<double>(  const char*, const char*, const std::vector<double>&  );
template void NP::Write<int>(     const char*, const char*, const std::vector<int>& );
template void NP::Write<unsigned>(const char*, const char*, const std::vector<unsigned>& );



inline void NP::WriteNames(
    const char* dir,
    const char* name,
    const std::vector<std::string>& names,
    unsigned num_names_,
    bool append )
{
    std::string _path = U::form_path(dir, name);
    const char* path = _path.c_str();
    WriteNames(path, names, num_names_, append  );
}


inline void NP::WriteNames(
    const char* dir,
    const char* reldir,
    const char* name,
    const std::vector<std::string>& names,
    unsigned num_names_,
    bool append )
{
    std::string _path = U::form_path(dir, reldir, name);
    const char* path = _path.c_str();
    WriteNames(path, names, num_names_, append );
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

inline void NP::WriteNames(
    const char* path,
    const std::vector<std::string>& names,
    unsigned num_names_,
    bool append )
{
    // if(names.size() == 0) return ;   DONT EARLY EXIT AS MORE REASONABLE TO TRUNCATE THE FILE WHEN THERE ARE NO NAMES
    int rc = U::MakeDirsForFile(path);
    if( rc != 0 ) std::cerr << "NP::WriteNames ERR creating dirs " << std::endl ;
    assert( rc == 0 );

    unsigned names_size = names.size() ;
    unsigned num_names = num_names_ == 0 ? names_size : num_names_ ;
    assert( num_names <= names_size );

    std::ios_base::openmode mode = std::ios::out|std::ios::binary ;
    if(append) mode |= std::ios::app ;

    std::ofstream stream(path, mode );
    for( unsigned i=0 ; i < num_names ; i++) stream << names[i] << std::endl ;
    stream.close();
}



inline void NP::WriteNames_Simple(
    const char* dir,
    const char* name,
    const std::vector<std::string>& names )
{
    std::string _path = U::form_path(dir, name);
    const char* path = _path.c_str();

    WriteNames_Simple(path, names );

}

inline void NP::WriteNames_Simple(
    const char* path,
    const std::vector<std::string>& names )
{
    int rc = U::MakeDirsForFile(path);
    if( rc != 0 ) std::cerr << "NP::WriteNames_Simple ERR creating dirs " << std::endl ;
    assert( rc == 0 );

    INT num_names = names.size();
    std::ios_base::openmode mode = std::ios::out|std::ios::binary ;
    std::ofstream fp(path, mode );
    for( INT i=0 ; i < num_names ; i++) fp << names[i] << std::endl ;
    fp.close();
}




inline void NP::WriteString(const char* dir, const char* name_, const char* ext, const std::string& str, bool append ) // static
{
    std::string name = U::form_name( name_, ext );
    std::string path_ = U::form_path(dir, name.c_str() );
    const char* path = path_.c_str();
    const char* xpath = U::Resolve(path);

    if(VERBOSE) std::cout
       << "NP::WriteString"
       << " path " << ( path ? path : "-" )
       << " xpath " << ( xpath ? xpath : "-" )
       << " str.size " << str.size()
       << std::endl
       ;

    std::ios_base::openmode mode = std::ios::out|std::ios::binary ;
    if(append) mode |= std::ios::app ;
    std::ofstream stream(xpath, mode );
    stream << str << std::endl ;
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
    std::string path = U::form_path(base, relp);
    NP* a = LoadFromTxtFile<T>( path.c_str());
    a->lpath = path ;
    return a ;
}


/**
NP::LoadFromTxtFile
----------------------

1. resolves spec_or_path into path
2. reads txt from the file into str
3. creates array with NP::LoadFromString

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

    const char* str = U::ReadString2(path);
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

inline void NP::Split(std::vector<std::string>& elems, const char* str, char delim)
{
    std::stringstream uss(str) ;
    std::string elem ;
    while(std::getline(uss,elem,delim)) elems.push_back(elem) ;
}
inline void NP::GetUnits(std::vector<std::string>& units ) // static
{
    Split(units, UNITS, ' ');
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
inline NP* NP::ZEROProp(T dscale)  // static
{
    NP* a = NP::LoadFromString<T>(R"(
    1.55     *eV    0.0
    15.5     *eV    0.0
)" );

   a->pscale(dscale, 0);
   return a ;
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
                << "NP::LoadFromString"
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
            const char* fstr = fields[i].c_str();
            if(U::ConvertsTo<T>(fstr))
            {
                value.push_back(U::To<T>(fstr)) ;
                line_column += 1 ;
            }
            else
            {
                if(!IsListed(other, fstr)) other.push_back(fstr);
            }
        }

        if( num_column == UNSET )
        {
            num_column = line_column ;
        }
        else if( line_column != num_column )
        {
            std::cerr
                << "NP::LoadFromString"
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
        if(*c == q) count += 1 ;
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

inline const char* NP::Resolve( const char* spec)  // TODO: rename or eliminate this as same as U::Resolve
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

#endif

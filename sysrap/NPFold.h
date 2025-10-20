#pragma once
/**
NPFold.h : collecting and persisting NP arrays keyed by relative paths
========================================================================

Primary Source Location in *np* repo (not *opticks*)
---------------------------------------------------------

+---------+------------------------------------------------------------------------+
| Action  |  Command                                                               |
+=========+========================================================================+
| Browse  | open https://github.com/simoncblyth/np/blob/master/NPFold.h            |
+---------+------------------------------------------------------------------------+
| Edit    | vi ~/np/NPFold.h                                                       |
+---------+------------------------------------------------------------------------+
| Test    | cd ~/np/tests ; ./NPFold_..._test.sh                                   |
+---------+------------------------------------------------------------------------+
| Copy    | cd ~/np ; ./cp.sh # when diff copies to  ~/opticks/sysrap/NPFold.h     |
+---------+------------------------------------------------------------------------+


Load/Save Modes
----------------

There are two load/save modes:

1. with index txt file "NPFold_index.txt" : the default mode
   in which the ordering of the keys are preserved

2. without index txt file : the ordering of keys/arrays
   follows the sorted order from U::DirList


Supporting NPFold within NPFold recursively
----------------------------------------------

For example "Materials" NPFold containing sub-NPFold for each material and at higher
level "Properties" NPFold containing "Materials" and "Surfaces" sub-NPFold.

A sub-NPFold of an NPFold is simply represented by a key in the
index that does not end with ".npy" which gets stored into ff vector.

Loading txt property files
----------------------------

Simple txt property files can also be loaded.
NB these txt property files are input only.

The NPFold keys still use the .npy extension
and saving the NPFold proceeds normally saving the arrays into
standard binary .npy and sidecars.


Former load_fts approach vs current load_dir
-----------------------------------------------

Former use of fts filesystem traversal with load_fts led
to loading a directory tree of .npy all into a single NPFold
Now with load_dir each directory is loading into an NPFold
so loading a directory tree creates a corresponding tree of NPFold.

Hid fts.h usage behind WITH_FTS as getting compilation error on Linux::

    /usr/include/fts.h:41:3: error: #error "<fts.h> cannot be used with -D_FILE_OFFSET_BITS==64"
     # error "<fts.h> cannot be used with -D_FILE_OFFSET_BITS==64"
       ^~~~~

**/

#include <string>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <csignal>
#include <cstdio>
#include <sys/types.h>

#ifdef WITH_FTS
#include <fts.h>
#endif

#include <cstring>
#include <errno.h>
#include <sstream>
#include <iomanip>

#include "NPX.h"

struct NPFold
{
    // PRIMARY MEMBERS : KEYS, ARRAYS, SUBFOLD

    std::vector<std::string> kk ;
    std::vector<const NP*>   aa ;

    std::vector<std::string> ff ;  // keys of sub-NPFold
    std::vector<NPFold*> subfold ;


    // METADATA FIELDS
    std::string               headline ;
    std::string               meta ;
    std::vector<std::string>  names ;
    const char*               savedir ;
    const char*               loaddir ;

    // nodata:true used for lightweight access to metadata from many arrays
    bool                      nodata ;
    bool                      verbose_ ;

    // [TRANSIENT FIELDS : NOT COPIED BY CopyMeta
    bool                      allowempty ;
    bool                      allowonlymeta ;
    bool                      skipdelete ;   // set to true on subfold during trivial concat
    NPFold*                   parent ;      // set by add_subfold
    // ]TRANSIENT FIELDS

    static constexpr const char INTKEY_PREFIX = 'f' ;
    static constexpr const int UNDEF = -1 ;
    static constexpr const bool VERBOSE = false ;
    static constexpr const bool ALLOWEMPTY = false ;
    static constexpr const bool ALLOWONLYMETA = false ;
    static constexpr const bool SKIPDELETE = false ;
    static constexpr NPFold*    PARENT = nullptr ;

    static constexpr const char* DOT_NPY = ".npy" ;  // formerly EXT
    static constexpr const char* DOT_TXT = ".txt" ;
    static constexpr const char* DOT_PNG = ".png" ;
    static constexpr const char* DOT_JPG = ".jpg" ;
    static constexpr const char* TOP = "/" ;
    static constexpr const char* INDEX = "NPFold_index.txt" ;
    static constexpr const char* META  = "NPFold_meta.txt" ;
    static constexpr const char* NAMES = "NPFold_names.txt" ;
    static constexpr const char* kNP_PROP_BASE = "NP_PROP_BASE" ;


    static bool IsNPY(const char* k);
    static bool IsTXT(const char* k);
    static bool IsPNG(const char* k);
    static bool IsJPG(const char* k);
    static bool HasSuffix( const char* k, const char* s );
    static bool HasPrefix( const char* k, const char* p );

    static const char* BareKey(const char* k);  // without .npy
    static std::string FormKey(const char* k, bool change_txt_to_npy );
    static std::string FormKey(const char* k );

    static bool    IsValid(const NPFold* fold);
    static NPFold* LoadIfExists(const char* base);
    static bool    Exists(const char* base);
    static NPFold* Load_(const char* base );
    static NPFold* LoadNoData_(const char* base );

    static const char* Resolve(const char* base_, const char* rel1_=nullptr, const char* rel2_=nullptr);
    static NPFold* Load(const char* base);
    static NPFold* Load(const char* base, const char* rel );
    static NPFold* Load(const char* base, const char* rel1, const char* rel2 );

    static NPFold* LoadNoData(const char* base);
    static NPFold* LoadNoData(const char* base, const char* rel );
    static NPFold* LoadNoData(const char* base, const char* rel1, const char* rel2 );


    static NPFold* LoadProp(const char* rel0, const char* rel1=nullptr );

    static int Compare(const NPFold* a, const NPFold* b );
    static std::string DescCompare(const NPFold* a, const NPFold* b );


    // CTOR
    NPFold();

    void set_verbose( bool v=true );
    void set_skipdelete( bool v=true );
    void set_allowempty( bool v=true );
    void set_allowonlymeta( bool v=true );

    void set_verbose_r( bool v=true );
    void set_skipdelete_r( bool v=true );
    void set_allowempty_r( bool v=true );
    void set_allowonlymeta_r( bool v=true );

    enum { SET_VERBOSE, SET_SKIPDELETE, SET_ALLOWEMPTY, SET_ALLOWONLYMETA } ;
    static int SetFlag_r(NPFold* nd, int flag, bool v);

private:
    void check_integrity() const ;
public:

    // [subfold handling
    NPFold*      add_subfold(char prefix=INTKEY_PREFIX);
    void         add_subfold(int ikey     , NPFold* fo, char prefix=INTKEY_PREFIX ); // integer key formatted with prefix

    static constexpr const char* _NPFold__add_subfold_ALLOW_DUPLICATE_KEY = "NPFold__add_subfold_ALLOW_DUPLICATE_KEY" ;
    void         add_subfold(const char* f, NPFold* fo );

    bool         has_zero_subfold() const ;
    int          get_num_subfold() const ;
    NPFold*      get_subfold(unsigned idx) const ;

    const char*  get_last_subfold_key() const ;
    const char*  get_subfold_key(unsigned idx) const ;
    int          get_subfold_idx(const char* f) const ;
    int          get_subfold_idx(const NPFold* fo) const ;

    int          get_key_idx(const char* k) const ;
    int          get_arr_idx(const NP* a) const ;


    const char*  get_subfold_key_within_parent() const ;

    void         get_treepath_(std::vector<std::string>& elem) const ;
    std::string  get_treepath(const char* k=nullptr) const ;
    static std::string Treepath(const NPFold* f);

    NPFold*      get_subfold(const char* f) const ;
    bool         has_subfold(const char* f) const ;


    void find_arrays_with_key(  std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key) const ;
    void find_arrays_with_key_r(std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key) const ;
    static void FindArraysWithKey_r(const NPFold* nd, std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key, int d);
    static std::string DescArraysAndPaths( std::vector<const NP*>& rr, std::vector<std::string>& tt );


    const NP*      find_array(const char* apath) const ;
    const NP*      find_array(const char* base, const char* name) const ;

    NPFold*        find_subfold_(const char* fpath) const  ;
    const NPFold*  find_subfold(const char* fpath) const  ;


    const void     find_subfold_with_all_keys(
        std::vector<const NPFold*>& subs,
        const char* keys,
        char delim=',' ) const ;

    const void     find_subfold_with_all_keys(
        std::vector<const NPFold*>& subs,
        std::vector<std::string>& names,
        const char* keys,
        char delim=',' ) const ;


    void get_subfold_with_intkey(std::vector<const NPFold*>& subs, char prefix) const ;
    bool all_subfold_with_intkey(char prefix) const ;
    void get_all_subfold_unique_keys(std::vector<std::string>& uks) const ;



    int concat_strict(std::ostream* out=nullptr);
    int concat(std::ostream* out=nullptr);
    NP* concat_(const char* k, std::ostream* out=nullptr);
    bool can_concat(std::ostream* out=nullptr) const ;

    int maxdepth() const ;
    static int MaxDepth_r(const NPFold* nd, int d);


    static constexpr const int MXD_NOLIMIT = 0 ;
    static int Traverse_r(
        const NPFold* nd,
        std::string nd_path,
        std::vector<const NPFold*>& folds,
        std::vector<std::string>& paths,
        int d,
        int mxd=MXD_NOLIMIT );

    static std::string FormSubPath(const char* base, const char* sub, char delim='/' );

    std::string desc_subfold(const char* top=TOP) const ;
    void find_subfold_with_prefix(
         std::vector<const NPFold*>& subs,
         std::vector<std::string>* subpaths,
         const char* prefix,
         int maxdepth ) const ;

    static std::string DescFoldAndPaths( const std::vector<const NPFold*>& subs, const std::vector<std::string>& subpaths );

    bool is_empty() const ;
    int total_items() const ;

    // ]subfold handling


    void add(int ikey, const NP* a, char prefix, int wid=3);
    void add( const char* k, const NP* a);
    void add_(const char* k, const NP* a);
    void set( const char* k, const NP* a);

    static void SplitKeys( std::vector<std::string>& elem , const char* keylist, char delim=',');
    static std::string DescKeys( const std::vector<std::string>& elem, char delim=',' );

    void clear();
private:
    void clear_(const std::vector<std::string>* keep);
    void clear_arrays(const std::vector<std::string>* keep);
public:
    void clear_subfold();
    void clear_only(  const char* clrlist=nullptr, bool copy=true, char delim=',');
    void clear_except(const char* keylist=nullptr, bool copy=true, char delim=',');
    void clear_except_(const std::vector<std::string>& keep, bool copy ) ;


    //OLD API: NPFold* copy( const char* keylist, bool shallow_array_copy, char delim=',' ) const ;
    NPFold* deepcopy(const char* keylist=nullptr, char delim=',' ) const ;
    NPFold* shallowcopy(const char* keylist=nullptr, char delim=',' ) const ;
private:
    // make private to find them all and switch to above form
    NPFold* copy(   bool shallow_array_copy, const char* keylist=nullptr, char delim=',' ) const ;
public:
    static NPFold* Copy(const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys );
    static void CopyMeta( NPFold* b , const NPFold* a );
    static void CopyArray(   NPFold* dst , const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys );
    static void CopySubfold( NPFold* dst , const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys );

    int count_keys( const std::vector<std::string>* keys ) const ;


    // single level (non recursive) accessors

    int num_items() const ;
    const char* get_key(unsigned idx) const ;
    const NP*   get_array(unsigned idx) const ;

    int find(const char* k) const ;
    bool has_key(const char* k) const ;
    bool has_all_keys(const char* keys, char delim=',') const ;
    bool has_all_keys(const std::vector<std::string>& qq) const ;
    int  count_keys(const std::vector<std::string>& qq) const ;

    const NP* get(const char* k) const ;
    NP*       get_(const char* k);


    const NP* get_optional(const char* k) const ;
    int   get_num(const char* k) const ;   // number of items in array
    void  get_counts( std::vector<std::string>* keys, std::vector<int>* counts ) const ;
    static std::string DescCounts(const std::vector<std::string>& keys, const std::vector<int>& counts );




    template<typename T> T    get_meta(const char* key, T fallback=0) const ;  // for T=std::string must set fallback to ""
    std::string get_meta_string(const char* key) const ;  // empty when not found

    template<typename T> void set_meta(const char* key, T value ) ;


    int save(const char* base, const char* rel, const char* name) ;
    int save(const char* base, const char* rel) ;
    int save(const char* base) ;
    int save_verbose(const char* base) ;

    int _save_local_item_count() const ;
    int _save_local_meta_count() const ;
    int _save(const char* base) ;

    int  _save_arrays(const char* base);
    void _save_subfold_r(const char* base);

    void load_array(const char* base, const char* relp);
    void load_subfold(const char* base, const char* relp);

#ifdef WITH_FTS
    static int FTS_Compare(const FTSENT** one, const FTSENT** two);
    int  no_longer_used_load_fts(const char* base) ;
#endif

    int  load_dir(const char* base) ;
    static constexpr const char* load_dir_DUMP = "NPFold__load_dir_DUMP" ;

    int  load_index(const char* base) ;
    static constexpr const char* load_index_DUMP = "NPFold__load_index_DUMP" ;

    int load(const char* base ) ;
    static constexpr const char* load_DUMP = "NPFold__load_DUMP" ;

    int load(const char* base, const char* rel0, const char* rel1=nullptr ) ;


    std::string descKeys() const ;
    std::string desc() const ;

    std::string desc_(int depth) const ;
    std::string descf_(int depth) const ;
    std::string desc(int depth) const ;

    std::string descMetaKVS() const ;
    void getMetaKVS(std::vector<std::string>* keys, std::vector<std::string>* vals, std::vector<int64_t>* stamps, bool only_with_stamp ) const ;
    int  getMetaNumStamp() const ;

    std::string descMetaKV() const ;
    void getMetaKV(std::vector<std::string>* keys, std::vector<std::string>* vals, bool only_with_profile ) const ;
    int  getMetaNumProfile() const ;

    void setMetaKV(const std::vector<std::string>& keys, const std::vector<std::string>& vals) ;

    static std::string Indent(int width);

    std::string brief() const ;
    std::string stats() const ;
    std::string smry() const ;

    // STATIC CONVERTERS

    static void Import_MIMSD(            std::map<int,std::map<std::string,double>>& mimsd, const NPFold* f );
    static NPFold* Serialize_MIMSD(const std::map<int,std::map<std::string,double>>& mimsd);
    static std::string Desc_MIMSD( const std::map<int,std::map<std::string,double>>& mimsd);


    // SUMMARIZE FOLD ARRAY COUNTS
    NP* subcount( const char* prefix ) const ;
    static constexpr const char* subcount_DUMP = "NPFold__subcount_DUMP" ;

    NP* submeta(const char* prefix, const char* column=nullptr ) const ;

    // TIMESTAMP/PROFILE COMPARISON USING SUBFOLD METADATA

    NPFold* substamp(  const char* prefix, const char* keyname) const ;
    static constexpr const char* substamp_DUMP = "NPFold__substamp_DUMP" ;

    NPFold* subprofile(const char* prefix, const char* keyname) const ;
    static constexpr const char* subprofile_DUMP = "NPFold__subprofile_DUMP" ;

    template<typename ... Args>
    NPFold* subfold_summary(const char* method, Args ... args_  ) const  ;
    static constexpr const char* subfold_summary_DUMP = "NPFold__subfold_summary_DUMP" ;

    template<typename F, typename T>
    NP* compare_subarrays(const char* key, const char* asym="a", const char* bsym="b", std::ostream* out=nullptr  );

    template<typename F, typename T>
    std::string compare_subarrays_report(const char* key, const char* asym="a", const char* bsym="b" );


    static void Subkey(std::vector<std::string>& ukey, const std::vector<const NPFold*>& subs );
    static void SubCommonKV(
        std::vector<std::string>& okey,
        std::vector<std::string>& ckey,
        std::vector<std::string>& cval,
        const std::vector<const NPFold*>& subs );

    static std::string DescCommonKV(
         const std::vector<std::string>& okey,
         const std::vector<std::string>& ckey,
         const std::vector<std::string>& cval );

};


inline bool NPFold::IsNPY(const char* k) { return HasSuffix(k, DOT_NPY) ; }
inline bool NPFold::IsTXT(const char* k) { return HasSuffix(k, DOT_TXT) ; }
inline bool NPFold::IsPNG(const char* k) { return HasSuffix(k, DOT_PNG) ; }
inline bool NPFold::IsJPG(const char* k) { return HasSuffix(k, DOT_JPG) ; }


inline bool NPFold::HasSuffix(const char* k, const char* s )
{
    return k && s && strlen(k) >= strlen(s) && strncmp( k + strlen(k) - strlen(s), s, strlen(s)) == 0 ;
}
inline bool NPFold::HasPrefix( const char* k, const char* p )
{
    return k && p && strlen(p) <= strlen(k) && strncmp(k, p, strlen(p)) == 0 ;
}

/**
NPFold::BareKey
----------------

For keys ending with DOT_NPY ".npy" or DOT_TXT ".txt"
this returns the key without the extension.

**/

inline const char* NPFold::BareKey(const char* k)
{
    char* bk = strdup(k);
    if(IsNPY(bk) || IsTXT(bk)) bk[strlen(bk)-4] = '\0' ;
    return bk ;
}


/**
NPFold::FormKey
-----------------

If added keys do not end with the DOT_NPY ".npy" then the DOT_NPY is added prior to collection.

Note that even when collecting arrays created from txt files, such as with SProp.h
where files would have no extension (or .txt extension) it is still appropriate
to add the DOT_NPY .npy  to the NPFold in preparation for subsequent saving
and for the simplicity of consistency.

Empty arrays with argument key ending with .txt are assumed
to be NPX::Holder placeholder arrays that act to carry vectors
of strings in the array names metadata. The extension is
swapped to .npy for more standard handling.

**/


inline std::string NPFold::FormKey(const char* k, bool change_txt_to_npy)
{
    bool is_npy = IsNPY(k);
    bool is_txt = IsTXT(k);

    std::stringstream ss ;

    if(change_txt_to_npy && is_txt)
    {
        const char* bk = BareKey(k) ;
        ss << bk << DOT_NPY ;
    }
    else
    {
        ss << k ;
        if(!is_npy) ss << DOT_NPY ;
    }


    std::string str = ss.str();
    return str ;
}

inline std::string NPFold::FormKey(const char* k)
{
    bool is_npy = IsNPY(k);
    std::stringstream ss ;
    ss << k ;
    if(!is_npy) ss << DOT_NPY ;
    std::string str = ss.str();
    return str ;
}


inline bool NPFold::IsValid(const NPFold* fold) // static
{
    return fold && !fold->is_empty() ;
}

inline NPFold* NPFold::LoadIfExists(const char* base) // static
{
    return Exists(base) ? Load(base) : nullptr ;
}

inline bool NPFold::Exists(const char* base) // static
{
    return NP::Exists(base, INDEX);
}
inline NPFold* NPFold::Load_(const char* base )
{
    if(base == nullptr) return nullptr ;
    NPFold* nf = new NPFold ;
    nf->load(base);
    return nf ;
}

/**
NPFold::LoadNoData_
--------------------

**/

inline NPFold* NPFold::LoadNoData_(const char* base_ )
{
    if(base_ == nullptr) return nullptr ;
    const char* base = NP::PathWithNoDataPrefix(base_);
    NPFold* nf = new NPFold ;
    nf->load(base);
    return nf ;
}

inline const char* NPFold::Resolve(const char* base_, const char* rel1_, const char* rel2_ )
{
    const char* base = U::Resolve(base_, rel1_, rel2_ );
    if(base == nullptr) std::cerr
        << "NPFold::Resolve"
        << " FAILED "
        << " base_ " << ( base_ ? base_ : "-" )
        << " rel1_ " << ( rel1_ ? rel1_ : "-" )
        << " rel2_ " << ( rel2_ ? rel2_ : "-" )
        << " POSSIBLY UNDEFINED ENVVAR TOKEN "
        << std::endl
        ;
    return base ;
}


inline NPFold* NPFold::Load(const char* base_)
{
    const char* base = Resolve(base_);
    return Load_(base);
}
inline NPFold* NPFold::Load(const char* base_, const char* rel_)
{
    const char* base = Resolve(base_, rel_);
    return Load_(base);
}
inline NPFold* NPFold::Load(const char* base_, const char* rel1_, const char* rel2_ )
{
    const char* base = Resolve(base_, rel1_, rel2_ );
    return Load_(base);
}



inline NPFold* NPFold::LoadNoData(const char* base_)
{
    const char* base = Resolve(base_);
    return LoadNoData_(base);
}
inline NPFold* NPFold::LoadNoData(const char* base_, const char* rel_)
{
    const char* base = Resolve(base_, rel_);
    return LoadNoData_(base);
}
inline NPFold* NPFold::LoadNoData(const char* base_, const char* rel1_, const char* rel2_ )
{
    const char* base = Resolve(base_, rel1_, rel2_ );
    return LoadNoData_(base);
}







inline NPFold* NPFold::LoadProp(const char* rel0, const char* rel1 )
{
    const char* base = getenv(kNP_PROP_BASE) ;
    NPFold* nf = new NPFold ;
    nf->load(base ? base : "/tmp", rel0, rel1 );
    return nf ;
}

inline int NPFold::Compare(const NPFold* a, const NPFold* b )
{
    int na = a->num_items();
    int nb = b->num_items();
    bool item_match = na == nb ;
    if(!item_match ) return -1 ;

    int mismatch = 0 ;
    for(int i=0 ; i < na ; i++)
    {
        const char* a_key = a->get_key(i);
        const char* b_key = b->get_key(i);
        const NP*   a_arr = a->get_array(i);
        const NP*   b_arr = b->get_array(i);

        bool key_match = strcmp(a_key, b_key) == 0 ;
        bool arr_match = NP::Memcmp(a_arr, b_arr) == 0 ;

        if(!key_match) mismatch += 1  ;
        if(!key_match) std::cout
            << "NPFold::Compare KEY_MATCH FAIL"
            << " a_key " << std::setw(40) << a_key
            << " b_key " << std::setw(40) << b_key
            << " key_match " << key_match
            << std::endl
            ;

        if(!arr_match) mismatch += 1 ;
        if(!arr_match) std::cout
            << "NPFold::Compare ARR_MATCH FAIL"
            << " a_arr " << std::setw(40) << a_arr->sstr()
            << " b_arr " << std::setw(40) << b_arr->sstr()
            << " arr_match " << arr_match
            << std::endl
            ;
    }
    if(mismatch > 0) std::cout << "NPFold::Compare mismatch " << mismatch << std::endl ;
    return mismatch ;
}




inline std::string NPFold::DescCompare(const NPFold* a, const NPFold* b )
{
    int na = a ? a->num_items() : -1 ;
    int nb = b ? b->num_items() : -1 ;
    bool item_match = na == nb ;

    std::stringstream ss ;
    ss << "NPFold::DescCompare"
       << " a " << ( a ? "Y" : "N" )
       << " b " << ( b ? "Y" : "N" )
       << std::endl
       << " na " << na
       << " nb " << nb
       << " item_match " << item_match
       << std::endl
       << " a.desc "
       << std::endl
       << ( a ? a->desc() : "-" )
       << " b.desc "
       << std::endl
       << ( b ? b->desc() : "-" )
       << std::endl
       ;
    std::string s = ss.str();
    return s;
}



// CTOR

inline NPFold::NPFold()
    :
    kk(),
    aa(),
    ff(),
    subfold(),
    headline(),
    meta(),
    names(),
    savedir(nullptr),
    loaddir(nullptr),
    nodata(false),
    verbose_(VERBOSE),
    allowempty(ALLOWEMPTY),
    allowonlymeta(ALLOWONLYMETA),
    skipdelete(SKIPDELETE),
    parent(PARENT)
{
    if(verbose_) std::cerr << "NPFold::NPFold" << std::endl ;
}


inline void NPFold::set_verbose( bool v )
{
     verbose_ = v ;
}
inline void NPFold::set_skipdelete( bool v )
{
    skipdelete = v ;
}
inline void NPFold::set_allowempty( bool v )
{
    allowempty = v ;
}
inline void NPFold::set_allowonlymeta( bool v )
{
    allowonlymeta = v ;
}



inline void NPFold::set_verbose_r( bool v )
{
    SetFlag_r(this, SET_VERBOSE, v);
}
inline void NPFold::set_skipdelete_r( bool v )
{
    SetFlag_r(this, SET_SKIPDELETE, v);
}
inline void NPFold::set_allowempty_r( bool v )
{
    SetFlag_r(this, SET_ALLOWEMPTY, v);
}
inline void NPFold::set_allowonlymeta_r( bool v )
{
    SetFlag_r(this, SET_ALLOWONLYMETA, v);
}



inline int NPFold::SetFlag_r(NPFold* nd, int flag, bool v)
{
    switch(flag)
    {
        case SET_VERBOSE   : nd->set_verbose(v)    ; break ;
        case SET_SKIPDELETE: nd->set_skipdelete(v) ; break ;
        case SET_ALLOWEMPTY: nd->set_allowempty(v) ; break ;
        case SET_ALLOWONLYMETA: nd->set_allowonlymeta(v) ; break ;
    }

    int tot_fold = 1 ;

    assert( nd->subfold.size() == nd->ff.size() );
    int num_sub = nd->subfold.size();
    for(int i=0 ; i < num_sub ; i++)
    {
        NPFold* sub = nd->subfold[i] ;
        int num = SetFlag_r( sub, flag, v );
        tot_fold += num ;
    }
    return tot_fold ;
}





/**
NPFold::check_integrity
--------------------------

check_integrity of key and array vectors and similarly for subfold (non-recursive)

**/

inline void NPFold::check_integrity() const
{
    assert( kk.size() == aa.size() );
    assert( ff.size() == subfold.size() );
}





// [ subfold handling

inline NPFold* NPFold::add_subfold(char prefix)
{
    int ikey = subfold.size();
    NPFold* sub = new NPFold ;
    add_subfold(ikey, sub, prefix );
    return sub ;
}

inline void NPFold::add_subfold(int ikey, NPFold* sub, char prefix )
{
    int wid = 3 ;
    std::string skey = U::FormNameWithPrefix(prefix, ikey, wid);
    add_subfold(skey.c_str(), sub );
}

/**
NPFold::add_subfold
--------------------

CAUTION : this simply collects keys and NPFold pointers
into vectors, NO COPYING IS DONE.
However, clearing the fold will delete arrays within the fold.
Because of this beware of stale input pointers after clearing.

* Regard subfold added to an NPFold to belong to the fold
* Do not do silly things like adding the same pointer more than once

**/


inline void NPFold::add_subfold(const char* f, NPFold* fo )
{
    if(fo == nullptr) return ;

    bool unique_f  = std::find( ff.begin(), ff.end(), f ) == ff.end() ;
    bool unique_fo = std::find( subfold.begin(), subfold.end(), fo ) == subfold.end() ;


    int ALLOW_DUPLICATE_KEY = U::GetEnvInt(_NPFold__add_subfold_ALLOW_DUPLICATE_KEY, 0 );

    if(!unique_f) std::cerr
       << "NPFold::add_subfold"
       << " ERROR repeated subfold key f[" << ( f ? f : "-" ) << "]"
       << " ff.size " << ff.size()
       << "[" << _NPFold__add_subfold_ALLOW_DUPLICATE_KEY << "] " << ALLOW_DUPLICATE_KEY
       << "\n"
       ;

    if( !unique_f && ALLOW_DUPLICATE_KEY == 0 )
    {
        assert( unique_f ) ;
        std::raise(SIGINT);   // release builds remove assert
    }

    if(!unique_fo) std::cerr
       << "NPFold::add_subfold"
       << " ERROR repeated subfold pointer with key f[" << ( f ? f : "-" ) << "]"
       << " subfold.size " << subfold.size()
       << "\n"
       ;
    assert( unique_fo ) ;



    ff.push_back(f); // subfold keys
    subfold.push_back(fo);

    if(fo->parent != nullptr)
    {
        std::string fo_treepath = fo->get_treepath();
        std::string fo_parent_treepath = fo->parent->get_treepath();
        std::string this_treepath = get_treepath();

        std::cerr
            << "NPFold::add_subfold "
            << " WARNING changing parent of added subfold fo \n"
            << " fo.treepath [" << fo_treepath << "]\n"
            << " fo.parent.treepath [" << fo_parent_treepath << "]\n"
            << " this.treepath [" << this_treepath << "]\n"
            << "\n"
            ;
    }
    assert( fo->parent == nullptr );
    fo->parent = this ;
}


inline bool NPFold::has_zero_subfold() const
{
    return 0 == get_num_subfold();
}
inline int NPFold::get_num_subfold() const
{
    assert( ff.size() == subfold.size() );
    return ff.size();
}
inline NPFold* NPFold::get_subfold(unsigned idx) const
{
    return idx < subfold.size() ? subfold[idx] : nullptr ;
}



inline const char* NPFold::get_last_subfold_key() const
{
    return ff.size() > 0 ? ff[ff.size()-1].c_str() : nullptr ;
}

inline const char* NPFold::get_subfold_key(unsigned idx) const
{
    return idx < ff.size() ? ff[idx].c_str() : nullptr ;
}
inline int NPFold::get_subfold_idx(const char* f) const
{
    size_t idx = std::distance( ff.begin(), std::find( ff.begin(), ff.end(), f )) ;
    return idx < ff.size() ? idx : UNDEF ;
}
inline int NPFold::get_subfold_idx(const NPFold* fo) const
{
    size_t idx = std::distance( subfold.begin(), std::find( subfold.begin(), subfold.end(), fo )) ;
    return idx < subfold.size() ? idx : UNDEF ;
}

inline int NPFold::get_key_idx(const char* k) const
{
    size_t idx = std::distance( kk.begin(), std::find( kk.begin(), kk.end(), k )) ;
    return idx < kk.size() ? idx : UNDEF ;
}
inline int NPFold::get_arr_idx(const NP* a) const
{
    size_t idx = std::distance( aa.begin(), std::find( aa.begin(), aa.end(), a )) ;
    return idx < aa.size() ? idx : UNDEF ;
}



inline const char* NPFold::get_subfold_key_within_parent() const
{
    int idx = parent ? parent->get_subfold_idx(this) : -1 ;
    return idx == -1 ? nullptr : parent->get_subfold_key(idx) ;
}

inline void NPFold::get_treepath_(std::vector<std::string>& elem) const
{
     const NPFold* n = this ;
     while(n)
     {
         const char* sk = n->get_subfold_key_within_parent() ;
         elem.push_back(sk ? sk : "");
         n = n->parent ;
     }
}
inline std::string NPFold::get_treepath(const char* k) const
{
    std::vector<std::string> elem ;
    get_treepath_(elem);
    std::reverse(elem.begin(), elem.end());
    std::stringstream ss ;
    int num_elem = elem.size();
    for(int i=0 ; i < num_elem ; i++ ) ss << elem[i] << ( i < num_elem - 1 ? "/" : "" ) ;
    if(k) ss << "/" << k ;
    std::string str = ss.str() ;
    return str ;
}

/**
NPFold::Treepath
-----------------

Disconnected fold has empty string "" treepath, see::

     TEST=subcopy ~/np/tests/NPFold_copy_test.sh

     NPFold::Treepath(zzz)   : {/z/zz}
     NPFold::Treepath(zzz_c) : {}


**/

inline std::string NPFold::Treepath(const NPFold* f)
{
    std::stringstream ss ;
    ss << "{" << ( f ? f->get_treepath() : "-" ) << "}" ;
    std::string str = ss.str() ;
    return str ;
}


inline NPFold* NPFold::get_subfold(const char* f) const
{
    int idx = get_subfold_idx(f) ;
    return idx == UNDEF ? nullptr : get_subfold(idx) ;
}
inline bool NPFold::has_subfold(const char* f) const
{
    int idx = get_subfold_idx(f) ;
    return idx != UNDEF ;
}


/**
NPFold::find_arrays_with_key
-----------------------------

Collect arrays and treepaths within this fold that have the query key.
Would normally expect either zero or one entries.

The query key has ".npy" appended if not already present.

**/

inline void NPFold::find_arrays_with_key(std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key) const
{
    std::string q = FormKey(q_key);
    for(int i=0 ; i < int(kk.size()) ; i++)
    {
        const char* k = kk[i].c_str();
        const NP* a = aa[i] ;
        bool qk_match = strcmp(q.c_str(), k) == 0 ;

        if(qk_match)
        {
            std::string t = get_treepath(k);
            rr.push_back(a);
            tt.push_back(t);
        }
    }
}

/**
NPFold::find_arrays_with_key_r
--------------------------------

Recursively collect arrays and treepaths within the entire tree of folders that
have the query key.

**/

inline void NPFold::find_arrays_with_key_r(std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key) const
{
    FindArraysWithKey_r(this, rr, tt, q_key, 0);
}

inline void NPFold::FindArraysWithKey_r(const NPFold* nd, std::vector<const NP*>& rr, std::vector<std::string>& tt, const char* q_key, int d)
{
    nd->find_arrays_with_key(rr, tt, q_key);
    for(int i=0 ; i < int(nd->subfold.size()) ; i++ ) FindArraysWithKey_r(nd->subfold[i], rr, tt, q_key, d+1 );
}

inline std::string NPFold::DescArraysAndPaths( std::vector<const NP*>& rr, std::vector<std::string>& tt ) // static
{
    assert( rr.size() == tt.size() );
    int num = rr.size() ;
    std::stringstream ss ;
    ss << "NPFold::DescArraysAndPaths num " << num << "\n" ;
    for(int i=0 ; i < num ; i++ )
    {
        const NP* r = rr[i] ;
        const char* t = tt[i].c_str() ;
        ss
           << std::setw(10) << r->sstr()
           << " : "
           << std::setw(30) << ( t ? t : "-" )
           << " : "
           << r
           << "\n"
           ;
    }
    std::string str = ss.str() ;
    return str ;
}





/**
NPFold::find_array
--------------------

0. split apath into base and name
1. find the subfold using *base*
2. get the array from the subfold

**/

inline const NP* NPFold::find_array(const char* apath) const
{
    std::string base = U::DirName(apath);
    std::string name = U::BaseName(apath);
    return find_array( base.c_str(), name.c_str()) ;
}

inline const NP* NPFold::find_array(const char* base, const char* name) const
{
    const NPFold* fold = find_subfold(base);
    return fold ? fold->get(name) : nullptr  ;
}

inline NPFold* NPFold::find_subfold_(const char* qpath) const
{
    const NPFold* f = find_subfold(qpath) ;
    return const_cast<NPFold*>(f) ;
}

/**
NPFold::find_subfold using full subfold qpath, start path is ""
----------------------------------------------------------------

0. recursively collects vectors of folds and paths
1. attempts to match the qpath with the vector or paths to get the idx
2. returns the subfold or nullptr if not found

**/

inline const NPFold* NPFold::find_subfold(const char* qpath) const
{
    std::vector<const NPFold*> folds ;
    std::vector<std::string>   paths ;
    Traverse_r( this, "",  folds, paths, 0, MXD_NOLIMIT  );
    size_t idx = std::distance( paths.begin(), std::find( paths.begin(), paths.end(), qpath ) ) ;

    if(VERBOSE)
    {
        std::cout
            << "NPFold::find_subfold"
            << " qpath[" << qpath << "]" << std::endl
            << " paths.size " << paths.size() << std::endl
            << " folds.size " << folds.size() << std::endl
            << " idx " << idx << std::endl
            ;

        for(unsigned i=0 ; i < paths.size() ; i++)
            std::cout << i << " [" << paths[i].c_str() << "]" << std::endl ;
    }
    return idx < paths.size() ? folds[idx] : nullptr ;
}




inline const void NPFold::find_subfold_with_all_keys(
    std::vector<const NPFold*>& subs,
    const char* keys_,
    char delim ) const
{
    int num_sub = get_num_subfold();
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = get_subfold(i) ;
        if(sub->has_all_keys(keys_, delim)) subs.push_back(sub) ;
    }
}

inline const void NPFold::find_subfold_with_all_keys(
    std::vector<const NPFold*>& subs,
    std::vector<std::string>&   names,
    const char* keys_,
    char delim ) const
{
    int num_sub = get_num_subfold();
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = get_subfold(i) ;
        const char* name = get_subfold_key(i) ;
        if(sub->has_all_keys(keys_, delim))
        {
            subs.push_back(sub) ;
            names.push_back(name) ;
        }
    }
}

/**
NPFold::get_subfold_with_intkey
---------------------------------

Examples of intkey with prefix 'f': f000 f001

**/

inline void NPFold::get_subfold_with_intkey(std::vector<const NPFold*>& subs, char prefix) const
{
    int num_sub = subfold.size();
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = subfold[i] ;
        const std::string& fk = ff[i] ;
        const char* fkk = fk.c_str();
        if( strlen(fkk) > 1 && fkk[0] == prefix && U::IsIntegerString(fkk+1) ) subs.push_back(sub);
    }
}
inline bool NPFold::all_subfold_with_intkey(char prefix) const
{
    std::vector<const NPFold*> subs ;
    get_subfold_with_intkey(subs, prefix);
    return subfold.size() == subs.size();
}

inline void NPFold::get_all_subfold_unique_keys(std::vector<std::string>& uk) const
{
    int num_sub = subfold.size();
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = subfold[i] ;
        int num_k = sub->kk.size();
        for(int j=0 ; j < num_k ; j++)
        {
            const std::string& k = sub->kk[j];
            if(std::find(uk.begin(), uk.end(), k) == uk.end()) uk.push_back(k);
        }
    }
}







/**
NPFold::concat_strict
----------------------

Concatenate common arrays from level 1 subfolds
into the top level fold.

Typically usage::

    fold->concat() ;
    fold->clear_subfold();

Invoking *clear_subfold* after *concat* tidies up
resources used for the sub arrays halving the memory
usage.

The trivial case of a single subfold is handled by simply
adding the subfold array pointers at top level and
invoking NPFold::set_skipdelete on the subfold to
prevent the active arrays from being deleted by clear_subfold.

Note that skipdelete is not called at top level, so the resources
will be released when the top level is cleared.

Future
~~~~~~~~

This approach is simple but transiently requires twice the end state memory.
Other more progressive approaches might avoid being so memory expensive.
Example of progressive approach would be to add an NPFold to another
one by one.

Actually for the application of combining arrays from multiple launches,
there may be little advantage with more involved approaches
as probably the number of launches will normally be 1 and sometimes 2 or 3.

**/

inline int NPFold::concat_strict(std::ostream* out)
{
    bool zero_sub = has_zero_subfold();
    if(out) *out << "NPFold::concat_strict zero_sub " << ( zero_sub ? "YES" : "NO " ) << "\n" ;
    if(zero_sub)
    {
        if(out) *out << "NPFold::concat_strict zero_sub TRIVIAL NOTHING TO CONCAT \n" ;
        return 0 ;
    }

    bool can = can_concat(out);
    if(!can)
    {
        std::cerr << "NPFold::concat_strict can_concat FAIL : problem with subfold ? return 1 \n";
        return 1 ;
    }


    int num_sub = subfold.size();
    const NPFold* sub0 = num_sub > 0 ? subfold[0] : nullptr  ;
    const std::vector<std::string>* kk0 = sub0 ? &(sub0->kk) : nullptr ;

    int num_k = kk0 ? kk0->size() : 0 ;

    if(out) *out
        << "NPFold::concat_strict"
        << " num_sub " << num_sub
        << " num_k " << num_k
        << "\n"
        ;

    for(int i=0 ; i < num_k ; i++)
    {
        const char* k = (*kk0)[i].c_str();
        NP* a = concat_(k, out );

        if(out) *out
            << "NPFold::concat_strict"
            << " k " << ( k ? k : "-" )
            << " a " << ( a ? a->sstr() : "-" )
            << "\n"
            ;

        add(k, a);
    }
    return 0 ;
}

inline int NPFold::concat(std::ostream* out)
{
    bool zero_sub = has_zero_subfold();
    if(out) *out << "NPFold::concat zero_sub " << ( zero_sub ? "YES" : "NO " ) << "\n" ;
    if(zero_sub)
    {
        if(out) *out << "NPFold::concat zero_sub TRIVIAL NOTHING TO CONCAT \n" ;
        return 0 ;
    }

    std::vector<std::string> uk ;
    get_all_subfold_unique_keys(uk);

    int num_uk = uk.size() ;

    if(out) *out
        << "NPFold::concat"
        << " subfold.size " << subfold.size()
        << " num_uk " << num_uk
        << "\n"
        ;


    for(int i=0 ; i < num_uk ; i++)
    {
        const char* k = uk[i].c_str();

        NP* a = concat_(k, out );

        if(out) *out
            << "NPFold::concat"
            << " k " << ( k ? k : "-" )
            << " a " << ( a ? a->sstr() : "-" )
            << "\n"
            ;

        add(k, a);
    }

    return 0 ;
}






/**
NPFold::concat_
----------------

Concatenates arrays with key *k* from all immediate subfold.

When there is only one subfold the concat is trivially
done by adding subfold arrays to this fold.
However in that situation the subfold must be
marked skipdelete to prevent clear_subfold which
is recommended after concat from deleting the
active top level array.

**/

inline NP* NPFold::concat_(const char* k, std::ostream* out)
{
    int num_sub = subfold.size();
    if( num_sub == 0 ) return nullptr ;

    NP* a = nullptr ;
    if( num_sub == 1 )
    {
        NPFold* sub0 = subfold[0] ;
        const NP* a0 = sub0->get(k);
        a = const_cast<NP*>(a0) ;
        sub0->set_skipdelete(true);
        if(out) *out << "NPFold::concat_ trivial concat set skipdelete on the single subfold \n" ;
    }
    else if( num_sub > 1 )
    {
        std::vector<const NP*> aa ;
        for(int i=0 ; i < num_sub ; i++)
        {
            const NPFold* sub = subfold[i] ;
            const NP* asub = sub->get(k);
            if(asub) aa.push_back(asub);
        }
        if(out) *out << "NPFold::concat_ non-trivial concat \n" ;
        a = NP::Concatenate(aa);
    }
    return a ;
}





/**
NPFold::can_concat
-------------------

Require:

1. two level tree of subfold, ie maxdepth is 1
2. integer string keys with f prefix (for python identifier convenience) giving the concat order
3. common array keys in all subfold::

    f000/[a.npy,b.npy,c.npy]
    f001/[a.npy,b.npy,c.npy]
    f002/[a.npy,b.npy,c.npy]

4. all common array keys must not be present within the top level folder


The trivial case of a single subfold is handled
simply by adding the subfold pointers to this
fold and calling NPFold::set_skipdelete on the
single subfold.

**/

inline bool NPFold::can_concat(std::ostream* out) const
{
    int d = maxdepth();
    if(out) *out << "NPFold::can_concat maxdepth " << d << "\n" ;
    if(d != 1) return false ;

    int num_sub = subfold.size();
    if(out) *out << "NPFold::can_concat num_sub " << num_sub << "\n" ;
    if(num_sub == 0) return false ;  // ACTUALLY : THATS TRIVIAL CASE

    char prefix = INTKEY_PREFIX ;
    bool all_intkey = all_subfold_with_intkey(prefix);
    if(out) *out << "NPFold::can_concat all_intkey " << ( all_intkey ? "YES" : "NO " )  << "\n" ;
    if(!all_intkey) return false ;

    const NPFold* sub0 = subfold[0] ;
    const std::vector<std::string>& kk0 = sub0->kk ;
    // reliance on the first sub being complete is not appropriate
    // need to collect unique keys from all subs
    // and combine as they are available

    int num_top_kk0 = count_keys(kk0);
    if(out) *out << "NPFold::can_concat num_top_kk0 " << num_top_kk0 << "\n" ;
    if(num_top_kk0 > 0) return false ;

    int sub_with_all_kk0 = 1 ;
    for(int i=1 ; i < num_sub ; i++) if(subfold[i]->has_all_keys(kk0)) sub_with_all_kk0 += 1 ;
    if(out) *out << "NPFold::can_concat sub_with_all_kk0 " << sub_with_all_kk0 << "\n" ;
    // HMM: when really slicing small for debug purposes it can happen that do not
    // get any hits from some slices

    bool can = sub_with_all_kk0 == num_sub ;
    if(out) *out << "NPFold::can_concat can " << ( can ? "YES" : "NO " )  << "\n" ;
    return can ;
}

/*
NPFold::maxdepth
-----------------

Maximum depth of the tree of subfold, 0 for a single node tree.

*/


inline int NPFold::maxdepth() const
{
    return MaxDepth_r(this, 0);
}

inline int NPFold::MaxDepth_r(const NPFold* nd, int d) // static
{
    assert( nd->subfold.size() == nd->ff.size() );
    int num_sub = nd->subfold.size();
    if(num_sub == 0) return d ;

    int mx = 0 ;
    for(int i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = nd->subfold[i] ;
        mx = std::max(mx, MaxDepth_r(sub, d+1));
    }
    return mx ;

}





/**
NPFold::Traverse_r
-------------------

Traverse starting from a single NPFold and proceeding
recursively to visit its subfold and their subfold and so on.
Collects all folds and paths in vectors, where the paths
are concatenated from the keys at each recursion level just
like a file system.

**/

inline int NPFold::Traverse_r(
     const NPFold* nd,
     std::string path,
     std::vector<const NPFold*>& folds,
     std::vector<std::string>& paths,
     int d,
     int mxd ) // static
{

    assert( nd->subfold.size() == nd->ff.size() );
    unsigned num_sub = nd->subfold.size();

    if(mxd == MXD_NOLIMIT || d <= mxd )
    {
        folds.push_back(nd);
        paths.push_back(path);
    }

    int tot_items = nd->num_items() ;

    for(unsigned i=0 ; i < num_sub ; i++)
    {
        const NPFold* sub = nd->subfold[i] ;
        std::string subpath = FormSubPath(path.c_str(), nd->ff[i].c_str(), '/' ) ;

        int num = Traverse_r( sub, subpath,  folds, paths, d+1, mxd );
        tot_items += num ;
    }
    return tot_items ;
}






inline std::string NPFold::FormSubPath(const char* base, const char* sub, char delim ) // static
{
    assert(sub) ; // base can be nullptr : needed for root, but sub must always be defined
    std::stringstream ss ;
    if(base && strlen(base) > 0) ss << base << delim ;
    ss << sub ;
    std::string s = ss.str();
    return s ;
}

/**
NPFold::desc_subfold
---------------------

Provides summary information for the subfold of this fold
acting as an index to the full details that follow for each
subfold and so on recursively.

**/

inline std::string NPFold::desc_subfold(const char* top)  const
{
    std::vector<const NPFold*> folds ;
    std::vector<std::string>   paths ;
    assert( folds.size() == paths.size() );

    int tot_items = Traverse_r( this, top,  folds, paths, 0, MXD_NOLIMIT );

    std::stringstream ss ;
    ss << " tot_items " << tot_items << std::endl ;
    ss << " folds " << folds.size() << std::endl ;
    ss << " paths " << paths.size() << std::endl ;
    for(int i=0 ; i < int(paths.size()) ; i++)
    {
        const NPFold* f = folds[i] ;
        const std::string& p = paths[i] ;
        ss << std::setw(3) << i
           << " [" << p << "] "
           << f->smry()
           << std::endl
           ;
    }

    if(nodata) ss << " NODATA " ;

    std::string s = ss.str();
    return s ;
}

/**
NPFold::find_subfold_with_prefix
--------------------------------

**/


inline void NPFold::find_subfold_with_prefix(
    std::vector<const NPFold*>& subs,
    std::vector<std::string>* subpaths,
    const char* prefix,
    int maxdepth ) const
{
    std::vector<const NPFold*> folds ;
    std::vector<std::string>   paths ;


    int tot_items = Traverse_r( this, TOP,  folds, paths, 0, maxdepth );

    assert( folds.size() == paths.size() );
    int num_paths = paths.size();

    bool dump = false ;

    if(dump)
    {
        std::cerr
            << "NPFold::find_subfold_with_prefix"
            << " prefix " << ( prefix ? prefix : "-" )
            << " maxdepth " << maxdepth
            << " TOP " << TOP
            << " folds.size " << folds.size()
            << " paths.size " << paths.size()
            << " tot_items " << tot_items
            << " nodata " << nodata
            << std::endl
            ;

        for(int i=0 ; i < num_paths ; i++) std::cerr
            << "[" << paths[i] << "]"
            << std::endl
            ;
    }

    if(nodata == false && tot_items == 0) return ;

    for(int i=0 ; i < num_paths ; i++)
    {
        const NPFold* f = folds[i] ;
        const char* p = paths[i].c_str() ;
        if(U::StartsWith(p, prefix))
        {
            subs.push_back(f);
            if(subpaths) subpaths->push_back(p);
        }
    }
}

inline std::string NPFold::DescFoldAndPaths( const std::vector<const NPFold*>& subs, const std::vector<std::string>& subpaths )
{
    assert( subs.size() == subpaths.size() );
    std::stringstream ss ;
    ss << "[NPFold::DescFoldAndPaths\n" ;
    for(int i=0 ; i < int(subs.size()) ; i++ )
    {
        const NPFold* sub = subs[i];
        const char* p = subpaths[i].c_str(); \
        ss
           << " sub " << ( sub ? "YES" : "NO " )
           << "  p  " << p
           << "\n"
           ;
    }
    ss << "]NPFold::DescFoldAndPaths\n" ;
    std::string str = ss.str() ;
    return str ;
}




inline bool NPFold::is_empty() const
{
    return total_items() == 0 ;
}

/**
NPFold::total_items
---------------------

Assuming that a NoData fold with some metadata
will have total_items greater than zero ?

**/

inline int NPFold::total_items() const
{
    std::vector<const NPFold*> folds ;
    std::vector<std::string>   paths ;

    int tot_items = Traverse_r( this, TOP,  folds, paths, 0, MXD_NOLIMIT );
    return tot_items ;
}


// ] subfold handling


inline void NPFold::add(int ikey, const NP* a, char prefix, int wid)
{
    std::string skey = U::FormNameWithPrefix(prefix, ikey, wid);
    add(skey.c_str(), a );
}


/**
NPFold::add
------------

CAUTION : NO ARRAY COPYING IS DONE,
this simply collects key and pointer into vectors,
but clearing will delete those arrays.

* regard everything added to NPFold to belong to the fold.
* input pointers will become stale after clear, dereferencing
  them will SIGSEGV (if you are lucky)

This approach is used as NPFold is intended to work
with very large multi-gigabyte arrays : so users
need to think carefully regarding memory management.

When *k* ends with ".txt" the key is changed to ".npy"
to simplify handling of empty NPX::Holder arrays.

Previously only did that for "a->is_empty()" but
as also need the change on find decided its simpler
to always do that.

**/

inline void NPFold::add(const char* k, const NP* a)
{
    if(a == nullptr) return ;
    bool change_txt_to_npy = true ;
    std::string key = FormKey(k, change_txt_to_npy );
    add_(key.c_str(), a );
}

/**
NPFold::add_
--------------

This lower level method does not add DOT_NPY to keys

NB the assertion that the key is not already present
avoids potential memory leak if were to just replace
the pointer loosing connection with that previous allocation


**/
inline void NPFold::add_(const char* k, const NP* a)
{
    if(verbose_)
    {
        std::string p = get_treepath(k);
        std::cerr << "NPFold::add_ [" << p << "]" << a << " " << a->sstr() << "\n" ;
    }

    int k_idx = get_key_idx(k);
    bool have_key_already = k_idx != UNDEF ;
    if(have_key_already) std::cerr
        << "NPFold::add_ FATAL : have_key_already [" << k << "]  k_idx[" << k_idx << "]\n"  << desc_(0) ;
    assert( !have_key_already );


    int a_idx = get_arr_idx(a);
    bool have_arr_already = a_idx != UNDEF ;
    if(have_arr_already) std::cerr
        << "NPFold::add_ FATAL : have_arr_already k[" << k << "] a_idx[" << a_idx << "]\n" << desc_(0) ;
    assert( !have_arr_already );

    kk.push_back(k);
    aa.push_back(a);
}





inline void NPFold::set(const char* k, const NP* a)
{
    int idx = find(k);
    if(idx == UNDEF)
    {
        add(k, a);
    }
    else
    {
        const NP* old_a = aa[idx] ;
        delete old_a ;
        aa[idx] = a ;
    }
}





/**
NPFold::SplitKeys
--------------------

FormKey which adds .npy if not already present is applied to form the elem
as that is done by NPFold::add

**/

inline void NPFold::SplitKeys( std::vector<std::string>& elem , const char* keylist, char delim) // static
{
    bool change_txt_to_npy = false ;

    std::stringstream ss;
    ss.str(keylist)  ;
    std::string s;
    while (std::getline(ss, s, delim)) elem.push_back(FormKey(s.c_str(),change_txt_to_npy));
}

inline std::string NPFold::DescKeys( const std::vector<std::string>& elem, char delim )
{
    int num_elem = elem.size();
    std::stringstream ss;
    for(int i=0 ; i < num_elem; i++)
    {
        ss << elem[i] ;
        if( i < num_elem - 1) ss << delim ;
    }
    std::string str = ss.str();
    return str ;
}






/**
NPFold::clear (clearing this fold and all subfold recursively)
----------------------------------------------------------------

**/
inline void NPFold::clear()
{
    if(verbose_)
    {
        std::string p = get_treepath();
        std::cerr << "NPFold::clear ALL [" << p << "]" << this << "\n" ;
    }
    clear_(nullptr);
}


/**
NPFold::clear_
----------------

NB: std::vector<NP*>::clear destructs the (NP*)
pointers but not the objects (the NP arrays) they point to.


This method is private as it must be used in conjunction with
NPFold::clear_except in order to to keep (key, array) pairs
of listed keys.

1. check_integrity (non-recursive)
2. each NP array with corresponding key not in the keep list is deleted
3. clears the kk and aa vectors
4. for each subfold call NPFold::clear on it and clear the subfold and ff vectors


HUH: CLEARS ARRAY POINTER VECTOR BUT DOES NOT DELETE
ARRAYS WITH KEYS IN THE KEEP LIST SO IT LOOSES
ARRAY POINTERS OF KEPT ARRAYS

THAT CAN ONLY WORK IF THOSE POINTERS WERE GRABBED PREVIOUSLY
AS DEMONSTRATED BY clear_except

**/

inline void NPFold::clear_(const std::vector<std::string>* keep)
{
    check_integrity();
    clear_arrays(keep);
    clear_subfold();
}




inline void NPFold::clear_arrays(const std::vector<std::string>* keep)
{
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i];
        const std::string& k = kk[i] ;
        bool listed = keep && std::find( keep->begin(), keep->end(), k ) != keep->end() ;
        if(!listed && !skipdelete)
        {
            if(verbose_)
            {
                std::string p = get_treepath(k.c_str());
                std::cerr << "NPFold::clear_arrays.delete[" << p << "]" << a << " " << a->sstr() << "\n";
            }
            delete a ;
        }
    }
    aa.clear();
    kk.clear();
}


/**
NPFold::clear_subfold
----------------------

CAUTION: after doing clear_subfold nullify any pointers to objects that
were added to the subfolds and that are deleted by the clear_subfold.
This avoids SIGSEGV from dereferencing those stale pointers.

**/

inline void NPFold::clear_subfold()
{
    if(verbose_)
    {
        std::string p = get_treepath();
        std::cerr << "NPFold::clear_subfold[" << p << "]" << this << "\n";
    }

    check_integrity();
    int num_sub = subfold.size() ;
    [[maybe_unused]] int num_ff = ff.size() ;
    assert( num_ff == num_sub ) ;

    for(int i=0 ; i < num_sub ; i++)
    {
        NPFold* sub = const_cast<NPFold*>(subfold[i]) ;
        sub->clear();
    }
    subfold.clear();
    ff.clear();       // folder keys
}


/**
NPFold::clear_except
-----------------------

Clears the folder but preserves the (key, array) pairs
listed in the keeplist of keys.

copy:false
    uses the old arrays

copy:true
    creates copies of the arrays that are kept


It is not so easy to do partial erase from vector
as the indices keep changing as elements are removed.
So take a simpler approach:

1. first copy keys and arrays identified by the *keeplist* into tmp_kk, tmp_aa
2. do a normal clear of all elements, which deletes
3. add copied tmp_aa tmp_kk back to the fold

NB that this means old pointers will be invalidated.
Unsure if that will be a problem.


Q: HOW TO USE THIS WITHOUT LEAKING IN COPY:TRUE AND FALSE MODES ?

HMM: need to delete whats cleared to avoid leak ? COMFIRM THIS


**/


inline void NPFold::clear_except_(const std::vector<std::string>& keep, bool copy )
{
    check_integrity();

    std::vector<const NP*>   tmp_aa ;
    std::vector<std::string> tmp_kk ;

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i];
        const std::string& k = kk[i] ;
        bool listed = std::find( keep.begin(), keep.end(), k ) != keep.end() ;
        if(listed)
        {
            tmp_aa.push_back(copy ? NP::MakeCopy(a) : a );
            tmp_kk.push_back(k);
        }
    }

    if(copy == true)
    {
        clear_(nullptr);  // remove all (k,a) pairs
    }
    else
    {
        clear_(&keep);    // remove all apart from the keep list, clears all keys
    }


    assert( tmp_aa.size() == tmp_kk.size() );
    for(unsigned i=0 ; i < tmp_aa.size() ; i++)
    {
        const NP* a = tmp_aa[i];
        const std::string& k = tmp_kk[i] ;
        add_( k.c_str(), a );
    }
}


inline void NPFold::clear_except(const char* keeplist, bool copy, char delim )
{
    if(verbose_) std::cerr
         << "NPFold::clear_except("
         << " keeplist:" << keeplist
         << " copy:" << copy
         << " delim:" << delim
         << ")"
         << std::endl
         ;

    std::vector<std::string> keep ;
    if(keeplist) SplitKeys(keep, keeplist, delim);

    clear_except_(keep, copy );
}

/**
NPFold::clear_only
-------------------

This is an alternative interface to clear_except which
forms a keeplist based on the keys present in the NPFold
and the ones on the clear list.

HMM: need to delete whats cleared to avoid leak ? COMFIRM THIS

**/

inline void NPFold::clear_only(const char* clearlist, bool copy, char delim )
{
    std::vector<std::string> clr ;
    if(clearlist) SplitKeys(clr, clearlist, delim);

    std::vector<std::string> keep ;
    int num_k = kk.size();
    for(int i=0 ; i < num_k ; i++)
    {
        const char* k = kk[i].c_str();
        bool k_listed = std::find( clr.begin(), clr.end(), k ) != clr.end() ;
        if(!k_listed) keep.push_back(k) ;
    }
    clear_except_(keep, copy );
}



inline NPFold* NPFold::deepcopy( const char* keylist, char delim ) const
{
    bool shallow_array_copy = false ;
    return copy(shallow_array_copy, keylist, delim);
}
inline NPFold* NPFold::shallowcopy( const char* keylist, char delim ) const
{
    bool shallow_array_copy = true ;
    return copy(shallow_array_copy, keylist, delim);
}


/**
NPFold::copy
---------------

Formerly returned nullptr when none of this folds keys
are specified in the keylist. However changed this
as sometimes want just the fold metadata.
A new NPFold is created and populated with any keylist
selected arrays from this fold.

shallow:true
    array pointers are copied as is

shallow:false
    arrays are copies and new array pointers used

NB the shallow refers to the arrays, not the NPFold that
are lightweight whose pointers are never copies as is

Also the keylist refer to array keys, not folder keys

**/

inline NPFold* NPFold::copy(bool shallow_array_copy, const char* keylist, char delim ) const
{
    std::vector<std::string> keys ;
    if(keylist) SplitKeys(keys, keylist, delim);
    // SplitKeys adds .npy to keys if not already present

    int count = count_keys(&keys) ;
    if( keylist && count == 0 && VERBOSE) std::cerr
        << "NPFold::copy"
        << " VERBOSE " << ( VERBOSE ? "YES" : "NO " )
        << " NOTE COUNT_KEYS GIVING ZERO "
        << " keylist [" << ( keylist ? keylist : "-" ) << "]"
        << " keylist.keys [" << DescKeys(keys, delim) << "]"
        << " count " << count
        << " kk.size " << kk.size()
        << " DescKeys(kk) [" << DescKeys(kk,',')  << "]"
        << " meta " << ( meta.empty() ? "EMPTY" : meta )
        << std::endl
        ;

    return NPFold::Copy(this, shallow_array_copy, keylist ? &keys : nullptr );
}


inline NPFold* NPFold::Copy(const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys ) // static
{
    src->check_integrity();
    NPFold* dst = new NPFold ;
    CopyMeta(dst, src);
    CopyArray(  dst, src, shallow_array_copy, keys );
    CopySubfold(dst, src, shallow_array_copy, keys );
    dst->check_integrity();
    return dst ;
}

/**
NPFold::CopyMeta
-----------------

Some members are not copied, namely::

    allowempty
    allowonlymeta
    skipdelete
    verbose_
    parent

**/

inline void NPFold::CopyMeta( NPFold* dst , const NPFold* src ) // static
{
    dst->headline = src->headline ;
    dst->meta = src->meta ;
    dst->names = src->names ;
    dst->savedir = src->savedir ? strdup(src->savedir) : nullptr ;
    dst->loaddir = src->loaddir ? strdup(src->loaddir) : nullptr ;
    dst->nodata  = src->nodata ;
    dst->verbose_ = src->verbose_ ;
}

/**
NPFold::CopyArray
------------------

keys:nullptr
   signals copy all arrays without selection

keys:!nullptr
   only arrays with listed keys are copied

**/

inline void NPFold::CopyArray( NPFold* dst , const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys ) // static
{
    for(int i=0 ; i < int(src->aa.size()) ; i++)
    {
        const NP* a = src->aa[i];
        const char* k = src->kk[i].c_str() ;
        bool listed = keys != nullptr && std::find( keys->begin(), keys->end(), k ) != keys->end() ;
        bool docopy = keys == nullptr || listed ;
        const NP* dst_a = docopy ? ( shallow_array_copy ? a : NP::MakeCopy(a) ) : nullptr  ;
        if(dst_a) dst->add_( k, dst_a );
    }

    if( keys == nullptr )
    {
        assert( src->aa.size() == dst->aa.size() ) ;
    }
}

inline void NPFold::CopySubfold( NPFold* dst , const NPFold* src, bool shallow_array_copy, std::vector<std::string>* keys ) // static
{
    for(int i=0 ; i < int(src->ff.size()) ; i++)
    {
        const char* k = src->ff[i].c_str() ;
        NPFold* fo = Copy(src->subfold[i], shallow_array_copy, keys);
        dst->add_subfold( k, fo );
    }
}






/**
NPFold::count_keys
------------------

Returns a count of immediate keys in the fold that are listed in the keys vector.

**/

inline int NPFold::count_keys( const std::vector<std::string>* keys ) const
{
    check_integrity();
    int count = 0 ;
    for(unsigned i=0 ; i < kk.size() ; i++)
    {
        const char* k = kk[i].c_str() ;
        bool listed = keys && std::find( keys->begin(), keys->end(), k ) != keys->end() ;
        if(listed) count += 1 ;
    }
    return count ;
}


// single level (non recursive) accessors

inline int NPFold::num_items() const
{
    check_integrity();
    return kk.size();
}


inline const char* NPFold::get_key(unsigned idx) const
{
    return idx < kk.size() ? kk[idx].c_str() : nullptr ;
}

inline const NP* NPFold::get_array(unsigned idx) const
{
    return idx < aa.size() ? aa[idx] : nullptr ;
}

/**
NPFold::find (non recursive)
-----------------------------

If the query key *k* does not end with the DOT_NPY ".npy" then that is added before searching.

std::find returns iterator to the first match

**/
inline int NPFold::find(const char* k) const
{
    bool change_txt_to_npy = true ;
    std::string key = FormKey(k, change_txt_to_npy);
    size_t idx = std::distance( kk.begin(), std::find( kk.begin(), kk.end(), key.c_str() )) ;
    return idx < kk.size() ? idx : UNDEF ;
}

inline bool NPFold::has_key(const char* k) const
{
    int idx = find(k);
    return idx != UNDEF  ;
}
inline bool NPFold::has_all_keys(const char* qq_, char delim) const
{
    std::vector<std::string> qq ;
    U::Split(qq_, delim, qq ) ;
    return has_all_keys(qq);
}

inline bool NPFold::has_all_keys(const std::vector<std::string>& qq) const
{
    int num_q = qq.size() ;
    int q_count = count_keys( qq );
    bool has_all = num_q > 0 && q_count == num_q ;
    return has_all ;
}

inline int NPFold::count_keys(const std::vector<std::string>& qq) const
{
    int num_q = qq.size() ;
    int q_count = 0 ;
    for(int i=0 ; i < num_q ; i++)
    {
       const char* q = qq[i].c_str() ;
       if(has_key(q)) q_count += 1 ;
    }
    return q_count ;
}


inline const NP* NPFold::get(const char* k) const
{
    int idx = find(k) ;
    return idx == UNDEF ? nullptr : aa[idx] ;
}

inline NP* NPFold::get_(const char* k)
{
    const NP* a = get(k) ;
    return const_cast<NP*>(a) ;
}


/**
NPFold::get_optional
---------------------

For now just the same as NPFold::get but in future
could assert that NPFold::get finds something whereas get_optional
is allowed to return nullptr.

**/
inline const NP* NPFold::get_optional(const char* k) const
{
    return get(k);
}





/**
NPFold::get_num
-----------------

Number of items in the array with key *k* or -1 if not such key.
**/

inline int NPFold::get_num(const char* k) const
{
    const NP* a = get(k) ;
    return a == nullptr ? UNDEF : a->shape[0] ;
}


inline void NPFold::get_counts( std::vector<std::string>* keys, std::vector<int>* counts ) const
{
    int nkk = kk.size();
    for(int i=0 ; i < nkk ; i++)
    {
        const char* k = kk[i].c_str();
        const NP* a = get(k) ;
        if(a == nullptr) continue ;
        if(keys) keys->push_back(k);
        if(counts) counts->push_back(a->shape[0]);
    }
}
inline std::string NPFold::DescCounts(const std::vector<std::string>& keys, const std::vector<int>& counts )
{
    assert( keys.size() == counts.size() );
    int num_key = keys.size();
    std::stringstream ss ;
    ss << "NPFold::DescCounts num_key " << num_key << std::endl ;
    for(int i=0 ; i < num_key ; i++ ) ss << std::setw(20) << keys[i] << " : " << counts[i] << std::endl ;
    std::string str = ss.str() ;
    return str ;
}







template<typename T> inline T NPFold::get_meta(const char* key, T fallback) const
{
    if(meta.empty()) return fallback ;
    return NP::GetMeta<T>( meta.c_str(), key, fallback );
}

template int         NPFold::get_meta<int>(const char*, int ) const ;
template unsigned    NPFold::get_meta<unsigned>(const char*, unsigned ) const  ;
template float       NPFold::get_meta<float>(const char*, float ) const ;
template double      NPFold::get_meta<double>(const char*, double ) const ;
template std::string NPFold::get_meta<std::string>(const char*, std::string ) const ;


/**
NPFold::get_meta_string
-------------------------

If the key is not found returns an empty string

**/
inline std::string NPFold::get_meta_string(const char* key) const
{
    bool meta_empty = meta.empty();
    if(meta_empty)
    {
        std::string tp = get_treepath();
        std::cerr
            << "NPFold::get_meta_string"
            << " meta_empty " << ( meta_empty ? "YES" : "NO " )
            << " key " << ( key ? key : "-" )
            << " treepath " << tp
            << "\n"
            ;
    }
    return meta_empty ? "" : NP::get_meta_string(meta, key);
}





template<typename T> inline void NPFold::set_meta(const char* key, T value)
{
    NP::SetMeta(meta, key, value);
}

template void     NPFold::set_meta<int>(const char*, int );
template void     NPFold::set_meta<unsigned>(const char*, unsigned );
template void     NPFold::set_meta<float>(const char*, float );
template void     NPFold::set_meta<double>(const char*, double );
template void     NPFold::set_meta<std::string>(const char*, std::string );






inline int NPFold::save(const char* base_, const char* rel) // not const as sets savedir
{
    std::string _base = U::form_path(base_, rel);
    const char* base = _base.c_str();
    return save(base);
}

inline int NPFold::save(const char* base_, const char* rel, const char* name) // not const as sets savedir
{
    std::string _base = U::form_path(base_, rel, name);
    const char* base = _base.c_str();
    return save(base);
}






/**
NPFold::save
--------------

ISSUE : repeated use of save for a fold with no .npy ie with only subfolds
never truncates the index, so it just keeps growing at every save.

FIXED THIS BY NOT EARLY EXITING NP::WriteNames when kk.size is zero
SO THE INDEX ALWAYS GETS TRUNCATED

**/

inline int NPFold::save(const char* base_)  // not const as calls _save
{
    const char* base = U::Resolve(base_);

    if(base == nullptr) std::cerr
        << "NPFold::save(\"" << ( base_ ? base_ : "-" ) << "\")"
        << " did not resolve all tokens in argument "
        << std::endl
        ;
    if(base == nullptr) return 1 ;

    return _save(base) ;
}

inline int NPFold::save_verbose(const char* base_)  // not const as calls _save
{
    const char* base = U::Resolve(base_);
    std::cerr
        << "NPFold::save(\"" << ( base_ ? base_ : "-" ) << "\")"
        << std::endl
        << " resolved to  [" << ( base ? base : "ERR-FAILED-TO-RESOLVE-TOKENS" ) << "]"
        << std::endl
        ;
    if(base == nullptr) return 1 ;
    return _save(base) ;
}


/**
NPFold::_save_local_item_count
--------------------------------

This is used by SEvt::_save to avoid writing
empty dirs

TODO: using NP directory making could allow to
encapsulate this within here

**/

inline int NPFold::_save_local_item_count() const
{
    return kk.size() + ff.size() ;
}

inline int NPFold::_save_local_meta_count() const
{
    bool with_meta = !meta.empty() ;
    bool with_names = names.size() > 0 ;
    return int(with_meta) + int(with_names) ;
}



/**
NPFold::_save
---------------

allowempty
    default from ALLOWEMPTY is false
    when true proceeds with saving even when no arrays
    [HMM: can probably remove this after adding allowonlymeta?]

allowonlymeta
    default from ALLOWONLYMETA is false
    when true proceeds with saving when a folder
    contains only metadata (ie no arrays or subfold)



onlymeta_proceed
    no arrays or subfold but has metadata and allowonlymeta:true

**/

inline int NPFold::_save(const char* base)  // not const as sets savedir
{
    assert( !nodata );

    int slic = _save_local_item_count();
    int slmc = _save_local_meta_count();

    bool slic_proceed = slic > 0 || ( slic == 0 && allowempty == true ) ;
    bool onlymeta_proceed = slic == 0 && slmc > 0 && allowonlymeta == true ;
    bool proceed = slic_proceed || onlymeta_proceed ;

    if(!proceed) return 1 ;


    savedir = strdup(base);

    _save_arrays(base);

    NP::WriteNames(base, INDEX, kk );

    NP::WriteNames(base, INDEX, ff, 0, true  ); // append:true : write subfold keys (without .npy ext) to INDEX

    _save_subfold_r(base);

    bool with_meta = !meta.empty() ;

    if(with_meta) U::WriteString(base, META, meta.c_str() );

    NP::WriteNames_Simple(base, NAMES, names) ;

    return 0 ;
}




inline int NPFold::_save_arrays(const char* base) // using the keys with .npy ext as filenames
{
    int count = 0 ;
    for(unsigned i=0 ; i < kk.size() ; i++)
    {
        const char* k = kk[i].c_str() ;
        const NP* a = aa[i] ;
        if( a == nullptr )
        {
            if(VERBOSE) std::cerr
                << "NPFold::_save_arrays"
                << " base " << base
                << " k " << k
                << " ERROR MISSING ARRAY FOR KEY "
                << std::endl
                ;
        }
        else
        {
            a->save(base, k );
            count += 1 ;
        }
    }
    // this motivated adding directory creation to NP::save
    return count ;
}

inline void NPFold::_save_subfold_r(const char* base)  // NB recursively called via NPFold::save
{
    assert( subfold.size() == ff.size() );
    for(unsigned i=0 ; i < ff.size() ; i++)
    {
        const char* f = ff[i].c_str() ;
        NPFold* sf = subfold[i] ;
        sf->save(base, f );
    }
}




/**
NPFold::load_array
--------------------

0. NP::Load for relp ending .npy otherwise NP::LoadFromTxtFile<double>
1. add the array using relp as the key

**/
inline void NPFold::load_array(const char* _base, const char* relp)
{
    bool is_nodata = NP::IsNoData(_base);
    bool is_npy = IsNPY(relp) ;
    bool is_txt = IsTXT(relp) ;


    NP* a = nullptr ;

    if(is_npy)
    {
        a = NP::Load(_base, relp) ;
    }
    else if(is_nodata)   // nodata mode only do nodata load of arrays
    {
        a = nullptr ;
    }
    else if(is_txt)
    {
        a = NP::LoadFromTxtFile<double>(_base, relp) ;
    }
    else
    {
        a = nullptr ;
    }
    if(a) add(relp,a ) ;
}

/**
NPFold::load_subfold
---------------------

**/

inline void NPFold::load_subfold(const char* _base, const char* relp)
{
    assert(!IsNPY(relp));
    add_subfold(relp,  NPFold::Load(_base, relp) ) ;
}



#ifdef WITH_FTS
inline int NPFold::FTS_Compare(const FTSENT** one, const FTSENT** two)
{
    return (strcmp((*one)->fts_name, (*two)->fts_name));
}


/**
NPFold::no_longer_used_load_fts
----------------------------------

This was formerly called by NPFold::load when the
base directory does not include an INDEX file.

It has been replaced by NPFold::load_dir
because of the structural difference between
loading with index and loading with fts.

fts traverses the directory tree and invokes NPFold::load_array when
meeting regular files or symbolic links.
See "man fts" and tests/fts_test.sh for background.

This fts approach resulted in a single NPFold
with keys containing slash.

Switching to NP::load_dir unifies the structure
with an NPFold for each directory.

**/

inline int NPFold::no_longer_used_load_fts(const char* base_)
{
    char* base = const_cast<char*>(base_);
    char* basepath[2] {base, nullptr};

    // NB fs is file system hierarchy, not just one directory
    FTS* fs = fts_open(basepath,FTS_COMFOLLOW|FTS_NOCHDIR,&FTS_Compare);
    if(fs == nullptr) return 1 ;

    FTSENT* node = nullptr ;
    while((node = fts_read(fs)) != nullptr)
    {
        switch (node->fts_info)
        {
            case FTS_D :    // directory being visited in preorder
                 break;
            case FTS_F  :   // regular file
            case FTS_SL :   // symbolic link
                {
                    char* relp = node->fts_path+strlen(base)+1 ;
                    load_array(base, relp) ;
                }
                break;
            default:
                break;
        }
    }
    fts_close(fs);
    return 0 ;
}
#endif

/**
NPFold::load_dir
-----------------

Loads directory-by-directory into separate NPFold
unlike load_fts that loads entire tree into a single NPFold.

Excluding files with names ending run_meta.txt avoids
loading metadata sidecar files as those are loaded whilst loading
the associated array eg run.npy

**/

inline int NPFold::load_dir(const char* _base)
{
    const char* base = nodata ? _base + 1 : _base ;

    int _DUMP = U::GetEnvInt(load_dir_DUMP , 0);

    if(_DUMP > 0) std::cout << "[" << load_dir_DUMP << " : [" << ( base ? base : "-" )  << "]\n" ;

    std::vector<std::string> names ;
    const char* ext = nullptr ;
    bool exclude = false ;
    bool allow_nonexisting = false ;

    U::DirList(names, base, ext, exclude, allow_nonexisting) ;
    if(names.size() == 0) return 1 ;

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str();
        int type = U::PathType(base, name) ;

        if( type == U::FILE_PATH && U::EndsWith(name, "_meta.txt"))
        {
            if(_DUMP > 0) std::cerr << "-NPFold::load_dir SKIP metadata sidecar " << name << std::endl ;
        }
        else if( type == U::FILE_PATH )
        {
            load_array(_base, name) ;
        }
        else if( type == U::DIR_PATH && U::StartsWith(name, "_"))
        {
            if(_DUMP > 0) std::cerr << "-NPFold::load_dir SKIP directory starting with _" << name << std::endl ;
        }
        else if( type == U::DIR_PATH )
        {
            load_subfold(_base, name);  // instanciates NPFold and add_subfold
        }
    }

    if(_DUMP > 0) std::cout << "]" << load_dir_DUMP << " : [" << ( base ? base : "-" )  << "]\n" ;

    return 0 ;
}


inline int NPFold::load_index(const char* _base)
{
    const char* base = nodata ? _base + 1 : _base ;
    int _DUMP = U::GetEnvInt(load_index_DUMP,0);
    if(_DUMP>0) std::cout << "[" << load_index_DUMP << " : [" << ( base ? base : "-" )  << "]\n" ;


    std::vector<std::string> keys ;
    NP::ReadNames(base, INDEX, keys );
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const char* key = keys[i].c_str() ;
        if(IsNPY(key))
        {
            load_array(_base, key );   // invokes *add* appending to kk and aa
        }
        else
        {
            load_subfold(_base, key);  // instanciates NPFold and add_subfold
        }
    }
    if(_DUMP>0) std::cout << "]" << load_index_DUMP << " : [" << ( base ? base : "-" )  << "]\n" ;
    return 0 ;
}

/**
NPFold::load
---------------

Typical persisted NPFold have index files
so the load_dir is not ordinarily used.

**/

inline int NPFold::load(const char* _base)
{
    nodata = NP::IsNoData(_base) ;  // _path starting with NP::NODATA_PREFIX eg '@'
    const char* base = nodata ? _base + 1 : _base ;

    int _DUMP = U::GetEnvInt(load_DUMP, 0);
    if(_DUMP>0) std::cout << "[" << load_DUMP << " " << U::FormatLog() << " : [" << ( base ? base : "-" )  << "]\n" ;


    bool exists = NP::Exists(base);
    if(!exists && _DUMP>0) std::cout << "NPFold::load non-existing base[" << ( base ? base : "-" ) << "]" << std::endl ;
    if(!exists) return 1 ;

    loaddir = strdup(base);
    bool has_meta = NP::Exists(base, META) ;
    if(has_meta) meta = U::ReadString( base, META );

    bool has_names = NP::Exists(base, NAMES) ;
    if(has_names) NP::ReadNames( base, NAMES, names );

    bool has_index = NP::Exists(base, INDEX) ;
    int rc = has_index ? load_index(_base) : load_dir(_base) ;

    if(_DUMP>0) std::cout << "]" << load_DUMP << " " << U::FormatLog() << " : [" << ( base ? base : "-" ) << " rc " << rc << "]\n" ;
    return rc ;
}
inline int NPFold::load(const char* base_, const char* rel0, const char* rel1)
{
    std::string base = U::form_path(base_, rel0, rel1);
    return load(base.c_str());
}

inline std::string NPFold::descKeys() const
{
    int num_key = kk.size() ;
    std::stringstream ss ;
    ss << "NPFold::descKeys"
       << " kk.size " << num_key
       ;
    for(int i=0 ; i < num_key ; i++) ss << " [" << kk[i] << "] " ;

    std::string str = ss.str();
    return str ;
}


inline std::string NPFold::desc() const
{
    std::stringstream ss ;
    ss << "[NPFold::desc" << std::endl ;
    if(!headline.empty()) ss << headline << std::endl ;
    ss << "NPFold::desc_subfold"  << std::endl ;
    ss << desc_subfold() ;
    ss << "NPFold::desc(0) "  << std::endl ;
    ss << desc(0) << std::endl ;
    ss << "]NPFold::desc" << std::endl ;
    std::string str = ss.str();
    return str ;
}

inline std::string NPFold::descMetaKVS() const
{
    return NP::DescMetaKVS(meta);
}

/**
NPFold::getMetaKVS
---------------------

Parses the NPFold::meta string into keys, vals and stamps
where stamp are defaulted to zero if the val do not look
like microssecond timestamps.

**/


inline void NPFold::getMetaKVS(
    std::vector<std::string>* keys,
    std::vector<std::string>* vals,
    std::vector<int64_t>* stamps,
    bool only_with_stamp ) const
{
    U::GetMetaKVS(meta, keys, vals, stamps, only_with_stamp );
}


/**
NPFold::getMetaNumStamp
--------------------------

**/

inline int NPFold::getMetaNumStamp() const
{
    std::vector<std::string> keys ;
    std::vector<int64_t>   stamps ;
    bool only_with_stamp = true ;
    getMetaKVS(&keys, nullptr, &stamps, only_with_stamp );
    assert( keys.size() == stamps.size() );
    int count = 0 ;
    for(int i=0 ; i < int(keys.size()) ; i++) count += stamps[i] == 0 ? 0 : 1 ;
    return count ;
}




inline std::string NPFold::descMetaKV() const
{
    return NP::DescMetaKV(meta);
}

inline void NPFold::getMetaKV(
    std::vector<std::string>* keys,
    std::vector<std::string>* vals,
    bool only_with_profile ) const
{
    NP::GetMetaKV(meta, keys, vals, only_with_profile );
}

/**
NPFold::getMetaNumProfile
--------------------------

Returns the number of meta (key,value) entries where the values contain timestamps.

**/

inline int NPFold::getMetaNumProfile() const
{
    std::vector<std::string> keys ;
    std::vector<std::string> vals ;
    bool only_with_profile = true ;
    getMetaKV(&keys, &vals, only_with_profile );
    assert( keys.size() == vals.size() );
    int count = keys.size() ;
    return count ;
}

inline void NPFold::setMetaKV(const std::vector<std::string>& keys, const std::vector<std::string>& vals)
{
    NP::SetMetaKV_( meta, keys, vals );
}

inline std::string NPFold::desc_(int depth) const
{
    std::stringstream ss ;
    ss << "[NPFold::desc_(" << depth << ")\n" ;
    ss << brief() << std::endl ;
    ss << descMetaKVS() << std::endl ;
    for(unsigned i=0 ; i < kk.size() ; i++)
    {
        const char* k = kk[i].c_str() ;
        const NP* a = aa[i] ;
        ss
           << std::setw(4) << i << " : "
           << ( a && a->nodata ? "ND " : "   " )
           << Indent(depth*10)
           << std::setw(20) << k
           << " : " << ( a ? a->sstr() : "-" )
           << std::endl
           ;
    }
    ss << "]NPFold::desc_(" << depth << ")\n" ;
    std::string str = ss.str();
    return str ;
}
inline std::string NPFold::descf_(int depth) const
{
    std::stringstream ss ;
    ss << "[NPFold::descf_( " << depth << ")\n" ;
    for(unsigned i=0 ; i < ff.size() ; i++)
    {
        const char* f = ff[i].c_str()  ;
        ss << std::endl << f << std::endl ;

        NPFold* sf = subfold[i] ;
        ss << sf->desc(depth+1) << std::endl ;
    }
    ss << "]NPFold::descf_( " << depth << ")\n" ;
    std::string str = ss.str();
    return str ;
}


inline std::string NPFold::desc(int depth) const
{
    std::stringstream ss ;
    ss << desc_(depth);
    ss << descf_(depth);
    std::string str = ss.str();
    return str ;
}

inline std::string NPFold::Indent(int width)  // static
{
    std::string s(width, ' ');
    return s ;
}

inline std::string NPFold::brief() const
{
    std::stringstream ss ;
    if(loaddir) ss << " loaddir:" << loaddir ;
    if(savedir) ss << " savedir:" << savedir ;
    ss << stats() ;
    std::string str = ss.str();
    return str ;
}

inline std::string NPFold::stats() const
{
    std::stringstream ss ;
    ss <<  " subfold " << subfold.size() ;
    ss << " ff " << ff.size() ;
    ss << " kk " << kk.size() ;
    ss << " aa " << aa.size() ;
    std::string str = ss.str();
    return str ;
}

inline std::string NPFold::smry() const
{
    int num_stamp = getMetaNumStamp() ;
    std::stringstream ss ;
    ss << " stamp:" << num_stamp ;
    std::string str = ss.str();
    return str ;
}


// STATIC CONVERTERS


inline void NPFold::Import_MIMSD( std::map<int,std::map<std::string, double>>& mimsd, const NPFold* f ) // static
{
    typedef std::map<std::string, double> MSD ;

    int num_items = f->num_items();
    for(int idx=0 ; idx < num_items ; idx++)
    {
        const char* cat = f->get_key(idx);
        int icat = U::To<int>(cat);
        const NP* a = f->get_array(idx) ;

        MSD& msd = mimsd[icat] ;
        NPX::Import_MSD(msd, a );
    }
}


inline NPFold* NPFold::Serialize_MIMSD(const std::map<int,std::map<std::string, double>>& mimsd ) // static
{
    NPFold* f = new NPFold ;

    typedef std::map<std::string, double> MSD ;
    typedef std::map<int, MSD> MIMSD ;

    MIMSD::const_iterator it = mimsd.begin();

    for(unsigned i=0 ; i < mimsd.size() ; i++)
    {
        int icat = it->first ;
        const char* cat = U::FormName(icat) ;
        const MSD& msd = it->second ;
        NP* a = NPX::Serialize_MSD( msd );
        f->add(cat, a );

        std::advance(it, 1);
    }
    return f;
}

inline std::string NPFold::Desc_MIMSD(const std::map<int, std::map<std::string, double>>& mimsd) // static
{
    std::stringstream ss ;
    ss << "NPFold::Desc_MIMSD" << std::endl ;

    typedef std::map<std::string, double> MSD ;
    typedef std::map<int, MSD> MIMSD ;

    MIMSD::const_iterator it = mimsd.begin();

    for(unsigned i=0 ; i < mimsd.size() ; i++)
    {
        int cat = it->first ;
        const MSD& msd = it->second ;
        ss
            << " cat " << cat
            << " msd.size " << msd.size()
            << std::endl
            << NPX::Desc_MSD(msd)
            << std::endl
            ;

        std::advance(it, 1);
    }
    std::string s = ss.str();
    return s ;
}


/**
NPFold::subcount
------------------

Collects arrays item counts from multiple subfold
into single array for easy analysis/plotting etc.
Typical use is for comparing genstep, hit, photon etc
counts between multiple events during test scans.


1. find subfold of this fold with the prefix argument, eg prefix "//A" finds A000 A001 A002 ...
2. get unique list of array keys (eg "hit", "photon", "genstep") from all subfold
3. create 2d array of shape (num_sub, num_ukey) with array counts for each sub
4. return the array of array counts in each subfold


**/

inline NP* NPFold::subcount( const char* prefix ) const
{
    // 1. find subfold with prefix
    std::vector<const NPFold*> subs ;
    std::vector<std::string> subpaths ;
    int maxdepth = 1 ;  // only one level

    find_subfold_with_prefix(subs, &subpaths,  prefix, maxdepth );
    assert( subs.size() == subpaths.size() );
    int num_sub = int(subs.size()) ;

    // 2. get unique list of array keys from all subfold
    std::set<std::string> s_keys ;
    for(int i=0 ; i < num_sub ; i++)
    {
        std::vector<std::string> keys ;
        subs[i]->get_counts(&keys, nullptr);
        std::transform(
            keys.begin(), keys.end(),
            std::inserter(s_keys, s_keys.end()),
            [](const std::string& obj) { return obj ; }
           );
    }
    std::vector<std::string> ukey(s_keys.begin(), s_keys.end()) ;
    int num_ukey = ukey.size() ;

    int ni = num_sub ;
    int nj = num_ukey ;

    // 3. create 2d array of shape (num_sub, num_ukey) with array counts for each sub
    NP* a = NP::Make<int>( ni, nj  );
    a->labels = new std::vector<std::string> ;
    a->names = subpaths ;

    int* aa = a->values<int>() ;

    for(int i=0 ; i < num_ukey ; i++)
    {
        const char* uk  = ukey[i].c_str() ;
        const char* _uk = IsNPY(uk) ? BareKey(uk) : uk ;
        a->labels->push_back(_uk);
    }

    int _DUMP = U::GetEnvInt(subcount_DUMP,0);

    if(_DUMP>0) std::cout << "[" << subcount_DUMP << "\n" ;
    if(_DUMP>0) std::cout <<  " num_ukey " << num_ukey << std::endl ;
    if(_DUMP>0) for(int i=0 ; i < num_ukey ; i++ ) std::cout << a->names[i] << std::endl ;

    for(int i=0 ; i < ni ; i++)
    {
        std::vector<std::string> keys ;
        std::vector<int> counts ;
        subs[i]->get_counts(&keys, &counts);
        assert( keys.size() == counts.size() );
        int num_key = keys.size();

        for(int j=0 ; j < nj ; j++)
        {
            const char* uk = ukey[j].c_str();
            int idx = std::distance( keys.begin(), std::find(keys.begin(), keys.end(), uk ) ) ;
            int count = idx < num_key ? counts[idx] : -1  ;
            aa[i*nj+j] = count ;

            if(_DUMP>0) std::cout
                << std::setw(20) << uk
                << " idx " << idx
                << " count " << count
                << std::endl
                ;
        }
    }
    if(_DUMP>0) std::cout << "]" << subcount_DUMP << "\n" ;
    return a ;
}




/**
NPFold::submeta
------------------

1. find subfolders with prefix
2. collect metadata (k,v) pairs with common values for all subs into ckey, cval and other keys into okey
3. form an array of shape (num_sub, 1 when column_key provided OR num_okey when not )

**/

inline NP* NPFold::submeta(const char* prefix, const char* column_key ) const
{
    // 1. find subfolders with prefix

    std::vector<const NPFold*> subs ;
    std::vector<std::string> subpaths ;
    int maxdepth = 1 ;  // only look one level down

    find_subfold_with_prefix(subs, &subpaths,  prefix, maxdepth );
    assert( subs.size() == subpaths.size() );

    // 2. collect metadata (k,v) pairs with common values for all subs into ckey, cval and other keys into okey
    std::vector<std::string> okey ;
    std::vector<std::string> ckey ;
    std::vector<std::string> cval ;
    SubCommonKV(okey, ckey, cval, subs );
    assert( ckey.size() == cval.size() );
    bool dump_common = false ;
    if(dump_common) std::cout << DescCommonKV(okey, ckey, cval);

    int column = std::distance( okey.begin(), std::find( okey.begin(), okey.end(), column_key ? column_key : "-" )) ;
    bool found_column = column < int(okey.size()) ;

    int num_subs = subs.size() ;
    int num_okey = okey.size() ;
    int ni = num_subs ;
    int nj = found_column ? 1 : num_okey ;

    NP* a = NP::Make<int64_t>( ni, nj );
    int64_t* aa = a->values<int64_t>() ;

    a->names = subpaths ;
    a->labels = new std::vector<std::string>( okey.begin(), okey.end() ) ;

    for(int i=0 ; i < ni ; i++)
    {
        const NPFold* sub = subs[i] ;
        for(int j=0 ; j < nj ; j++)
        {
            const char* ok = found_column ? column_key : okey[j].c_str() ;
            int64_t val = sub->get_meta<int64_t>( ok, 0 );
            aa[i*nj+j] = val ;
        }
    }
    return a ;
}


/**
NPFold::substamp
--------------------

This provides metadata across multiple events, but as it relies on
saving of arrays it is not useful for production running because
SEvt are not saved because they are too big.

For metadata in production running the alternative SProf.hh
low resource approach should be used.

Example arguments:

* prefix "//A" "//B"
* keyname : "substamp"


Primary use of substamp is for comparisons of timestamp difference from begin of event
between multiple events eg A000 A001

1. finds *subs* vector of subfold of this fold with the path prefix, eg "//A" "//B"

2. create *t* array shaped (num_sub, num_stamp) containing timestamp values of the common keys,
   this is particularly useful when scanning with a sequence of
   events with increasing numbers of photons

3. derive *dt* DeltaColumn array, creating first-timestamp-within-each-event-relative-timestamps

4. create array of array counts (eg num_hit, num_genstep, num_photon) in each subfold

5. form *out* NPFold with keys "substamp" "delta_substamp" "subcount" containing the above created arrays



Example of NPFold_meta.txt::

    A[blyth@localhost ALL1_Debug_Philox_ref1]$ cat A000/NPFold_meta.txt

    NumPhotonCollected:1000000
    NumGenstepCollected:10
    MaxBounce:63

    site:SEvt::endMeta
    hitmask:8192
    index:0
    instance:0
    SEvt__beginOfEvent_0:1760707886287045,7316444,1222084
    SEvt__beginOfEvent_1:1760707886287165,7316444,1222084
    SEvt__endOfEvent_0:1760707886541450,8373000,1334844
    t_BeginOfEvent:1760707886287057
    t_setGenstep_0:0
    t_setGenstep_1:0
    t_setGenstep_2:0
    t_setGenstep_3:1760707886287216
    t_setGenstep_4:1760707886287368
    t_setGenstep_5:1760707886287387
    t_setGenstep_6:1760707886287407
    t_setGenstep_7:1760707886288094
    t_setGenstep_8:1760707886288391
    t_PreLaunch:1760707886288420
    t_PostLaunch:1760707886441593
    t_EndOfEvent:1760707886541457
    t_Event:254400
    t_Launch:0.153154


The first few of the above entries are written from SEvt::beginOfEvent with SEvt::setMeta.
Entries from "site:SEvt::endMeta" onwards are written from SEvt::endOfEvent/SEvt::endMeta
with::

    SEvt::setMeta
    SEvt::setMetaProf

Both the above methods append to the SEvt::meta string.
SEvt::meta is assigned to the NPFold by SEvt::gather_metadata
from SEvt::endOfEvent.

**/

inline NPFold* NPFold::substamp(const char* prefix, const char* keyname) const
{
    // 1. finds *subs* vector of subfold of this fold with the path prefix, eg "//A" "//B"

    std::vector<const NPFold*> subs ;
    std::vector<std::string> subpaths ;
    int maxdepth = 1 ;  // only one level down
    find_subfold_with_prefix(subs, &subpaths,  prefix, maxdepth );
    assert( subs.size() == subpaths.size() );
    int num_sub = int(subs.size()) ;

    int _DUMP = U::GetEnvInt(substamp_DUMP, 0 );

    const NPFold* sub0 = num_sub > 0 ? subs[0] : nullptr ;

    int num_stamp0 = sub0 ? sub0->getMetaNumStamp() : 0 ;
    bool skip = num_sub == 0 || num_stamp0 == 0 ;

    if(_DUMP) std::cout
        << "[" << substamp_DUMP
        << " find_subfold_with_prefix " << prefix
        << " maxdepth " << maxdepth
        << " num_sub " << num_sub
        << " sub0 " << ( sub0 ? sub0->stats() : "-" )
        << " num_stamp0 " << num_stamp0
        << " skip " << ( skip ? "YES" : "NO ")
        << std::endl
        << DescFoldAndPaths(subs, subpaths)
        ;

    NPFold* out = nullptr ;
    if(skip) return out ;


    // 2. create *t* array shaped (num_sub, num_stamp) containing timestamp values of the common keys

    int ni = num_sub ;
    int nj = num_stamp0 ; // num stamps in the first sub

    NP* t = NP::Make<int64_t>( ni, nj ) ;
    int64_t* tt = t->values<int64_t>() ;
    t->set_meta<std::string>("creator","NPFold::substamp");
    t->set_meta<std::string>("base", loaddir ? loaddir : "-" );
    t->set_meta<std::string>("prefix", prefix ? prefix : "-" );
    t->set_meta<std::string>("keyname", keyname ? keyname : "-" );

    // collect metadata (k,v) pairs that are the same for all the subs
    std::vector<std::string> okey ;
    std::vector<std::string> ckey ;
    std::vector<std::string> cval ;
    SubCommonKV(okey, ckey, cval, subs );
    assert( ckey.size() == cval.size() );
    t->setMetaKV_(ckey, cval);


    std::vector<std::string> comkeys ;
    for(int i=0 ; i < ni ; i++)
    {
        const NPFold* sub = subs[i] ;
        const char* subpath = subpaths[i].c_str() ;
        std::vector<std::string> keys ;
        std::vector<int64_t>   stamps ;


        // grab keys and stamps from the sub meta string
        bool only_with_stamps = true ;
        sub->getMetaKVS(&keys, nullptr, &stamps, only_with_stamps );

        int num_stamp = stamps.size() ;
        bool consistent_num_stamp = num_stamp == nj ;

        if(!consistent_num_stamp) std::cerr
            << "NPFold::substamp"
            << " i " << i
            << " subpath " << ( subpath ? subpath : "-" )
            << " consistent_num_stamp " << ( consistent_num_stamp ? "YES" : "NO " )
            << " num_stamp " << num_stamp
            << " nj " << nj
            << std::endl
            ;
        assert(consistent_num_stamp) ;

        if(i == 0) comkeys = keys ;
        bool same_keys = i == 0 ? true : keys == comkeys ;
        if(_DUMP>0) std::cout << sub->loaddir << " stamps.size " << stamps.size() << " " << ( same_keys ? "Y" : "N" ) << std::endl;
        assert(same_keys);

        for(int j=0 ; j < nj ; j++) tt[i*nj+j] = stamps[j] ;
        t->names.push_back(subpath);

    }
    t->labels = new std::vector<std::string>(comkeys.begin(), comkeys.end())  ;


    // 3. derive *dt* DeltaColumn array, creating first-timestamp-within-each-event-relative-timestamps

    NP* dt = NP::DeltaColumn<int64_t>(t);
    dt->names = t->names ;
    dt->labels = new std::vector<std::string>(comkeys.begin(), comkeys.end())  ;


    // 4. create array of array counts (eg num_hit, num_genstep, num_photon) in each subfold
    NP* count = subcount(prefix); // prefix eg "//A"


    // 5. form NPFold with keys "substamp" "delta_substamp" "subcount" containing the above created arrays

    const char* delta_keyname = U::FormName("delta_",keyname,nullptr) ; // normally "delta_substamp"
    out = new NPFold ;
    out->add(keyname      , t );  // "substamp"
    out->add(delta_keyname, dt ); // "delta_substamp"
    out->add("subcount", count );

    if(_DUMP>0) std::cout
        << "]" << substamp_DUMP
        << "\n"
        ;

    return out ;
}



/**
NPFold::subprofile
--------------------

Collect profile metadata from subfold matching the prefix

1. find *subs* vector of subfold of this fold with path prefix, eg "//A" "//B"
2. create *t* array of shape (num_sub, num_prof0, 3) with the profile triplets
3. create *out* NPFold containing "subprofile" keyname with the *t* array

**/

inline NPFold* NPFold::subprofile(const char* prefix, const char* keyname) const
{
    // 1. find *subs* vector of subfold of this fold with path prefix, eg "//A" "//B"

    std::vector<const NPFold*> subs ;
    std::vector<std::string> subpaths ;
    int maxdepth = 1 ;  // only one level down
    find_subfold_with_prefix(subs, &subpaths,  prefix, maxdepth );
    assert( subs.size() == subpaths.size() );
    int num_sub = int(subs.size()) ;
    int num_prof0 = num_sub > 0 ? subs[0]->getMetaNumProfile() : 0 ;
    bool skip = num_sub == 0 || num_prof0 == 0 ;

    int _DUMP = U::GetEnvInt(subprofile_DUMP, 0 );
    if(_DUMP>0) std::cout
        << "[" << subprofile_DUMP
        << " find_subfold_with_prefix " << prefix
        << " maxdepth " << maxdepth
        << " num_sub " << num_sub
        << " num_prof0 " << num_prof0
        << " skip " << ( skip ? "YES" : "NO ")
        << std::endl
        ;

    NPFold* out = nullptr ;
    if(skip) return out ;

    // 2. create *t* array of shape (num_sub, num_prof0, 3) with the profile triplets

    int ni = num_sub ;
    int nj = num_prof0 ;
    int nk = 3 ;

    NP* t = NP::Make<int64_t>( ni, nj, nk ) ;
    int64_t* tt = t->values<int64_t>() ;
    t->set_meta<std::string>("creator","NPFold::subprofile");
    t->set_meta<std::string>("base", loaddir ? loaddir : "-" );
    t->set_meta<std::string>("prefix", prefix ? prefix : "-" );
    t->set_meta<std::string>("keyname", keyname ? keyname : "-" );

    // collect metadata (k,v) pairs that are the same for all the subs
    std::vector<std::string> okey ;
    std::vector<std::string> ckey ;
    std::vector<std::string> cval ;
    SubCommonKV(okey, ckey, cval, subs );
    assert( ckey.size() == cval.size() );
    t->setMetaKV_(ckey, cval);

    std::vector<std::string> comkeys ;
    for(int i=0 ; i < ni ; i++)
    {
        const NPFold* sub = subs[i] ;
        const char* subpath = subpaths[i].c_str() ;

        if(_DUMP>0) std::cout
            << subpath
            << std::endl
            << sub->descMetaKV()
            << std::endl
            ;

        std::vector<std::string> keys ;
        std::vector<std::string> vals ;
        bool only_with_profiles = true ;
        sub->getMetaKV(&keys, &vals, only_with_profiles );
        assert( vals.size() == keys.size() ) ;
        assert( int(vals.size()) == nj ) ;

        if(i == 0) comkeys = keys ;
        bool same_keys = i == 0 ? true : keys == comkeys ;
        if(_DUMP>0) std::cout
             << "sub.loaddir " << sub->loaddir
             << " keys.size " << keys.size()
             << " " << ( same_keys ? "Y" : "N" )
             << std::endl
             ;
        assert(same_keys);

        for(int j=0 ; j < nj ; j++)
        {
            const char* v = vals[j].c_str();
            std::vector<int64_t> elem ;
            U::MakeVec<int64_t>( elem, v, ',' );
            assert( int(elem.size()) == nk );
            for(int k=0 ; k < nk ; k++)  tt[i*nj*nk+j*nk+k] = elem[k] ;
        }
        t->names.push_back(subpath);
    }
    t->labels = new std::vector<std::string>(comkeys.begin(), comkeys.end())  ;

    // 3. create *out* NPFold containing "subprofile" keyname with the *t* array

    out = new NPFold ;
    out->add(keyname, t );

    if(_DUMP>0) std::cout
        << "]" << subprofile_DUMP
        << std::endl
        ;
    return out ;
}



/**
NPFold::subfold_summary
-----------------------

Applies methods to each subfold found within this NPFold specified by k:v delimited argument values.
This creates summary sub or arrays for each group of subfold specified by the argument paths.

1. collect args containing ':' delimiter into uargs
2. create NPFold/NP for each uarg using *method* named arg , thats added to (NPFold)spec_ff
3. return (NPFold)spec_ff


Supported *method* are:

substamp

subprofile

submeta
   forms array of shape (num_sub, num_okey) with entries for each sub
   providing all non-common metadata values for each sub

submeta:some-column
   forms array of shape (num_sub, 1) with the some-column values for each sub

subcount




Example arguments::

   NPFold* ab = NPFold::subfold_summary("substamp",   "a://A", "b://B" ) ;
   NPFold* ab = NPFold::subfold_summary("subprofile", "a://A", "b://B" ) ;
   NPFold* ab = NPFold::subfold_summary("submeta",    "a://A", "b://B" ) ;
   NPFold* ab = NPFold::subfold_summary("subcount",   "a://A", "b://B" ) ;

**/

template<typename ... Args>
inline NPFold* NPFold::subfold_summary(const char* method, Args ... args_  ) const
{
    int _DUMP = U::GetEnvInt( subfold_summary_DUMP, 0 );


    // 1. collect args containing ':' delimiter into uargs

    std::vector<std::string> args = {args_...};
    std::vector<std::string> uargs ;
    char delim = ':' ;
    for(int i=0 ; i < int(args.size()) ; i++)
    {
        const std::string& arg = args[i] ;
        size_t pos = arg.empty() ? std::string::npos : arg.find(delim) ;
        if( pos == std::string::npos ) continue ;
        uargs.push_back( arg );
    }
    int num_uargs = uargs.size() ;
    if(_DUMP > 0)
    {
        std::cerr
           << "@[" << subfold_summary_DUMP
           << " method [" << ( method ? method : "-" ) << "]"
           << " args.size " << args.size()
           << " uargs.size " << uargs.size()
           << " uargs("
           ;

        for(int i=0 ; i < num_uargs ; i++) std::cerr << uargs[i] << " " ;
        std::cerr << ")\n" ;
    }


    std::stringstream hh ;
    hh << "NPFold::subfold_summary(\"" << method << "\"," ;

    // 2. create NPFold/NP for each argument using *method* named argument, thats added to (NPFold)spec_ff

    NPFold* spec_ff = nullptr ;

    for(int i=0 ; i < num_uargs ; i++)
    {
        const std::string& arg = uargs[i] ;
        hh << "\"" << arg << "\"" << ( i < num_uargs - 1 ? "," : " " ) ;

        size_t pos = arg.find(delim) ;
        std::string _k = arg.substr(0, pos);
        std::string _v = arg.substr(pos+1);
        const char* k = _k.c_str();   // "a" OR "b"
        const char* v = _v.c_str();   // eg "//A" "//B"


        NPFold* sub = nullptr ;
        NP* arr = nullptr ;

        if(strcmp(method, "substamp")==0)
        {
            sub = substamp(v, "substamp") ;
        }
        else if(strcmp(method, "subprofile")==0)
        {
            sub = subprofile(v, "subprofile") ;
        }
        else if(strcmp(method, "submeta")==0)
        {
            arr = submeta(v) ;
        }
        else if(strcmp(method, "subcount")==0)
        {
            arr = subcount(v) ;
        }
        else if(U::StartsWith(method, "submeta:"))
        {
            arr = submeta(v, method+strlen("submeta:") );
        }

        if(sub == nullptr && arr == nullptr)
        {
            if( _DUMP > 0 ) std::cerr
                << "@-NPFold::subfold_summary"
                << " method [" << ( method ? method : "-" ) << "]"
                << " k [" << k << "]"
                << " v [" << v << "]"
                << " sub " << ( sub ? "YES" : "NO " )
                << " arr " << ( arr ? "YES" : "NO " )
                << std::endl
                ;

            continue ;
        }
        if(spec_ff == nullptr) spec_ff = new NPFold ;
        if(sub) spec_ff->add_subfold(k, sub );
        if(arr) spec_ff->add(k, arr) ;
        // k does not stomp : as those are different spec_ff
        // HUH: looks to be same spec_ff - the k must be different to avoid stomping
    }
    hh << ")" ;

    if(spec_ff) spec_ff->headline = hh.str();
    if(_DUMP > 0) std::cerr
        << "@[" << subfold_summary_DUMP
        << " method [" << ( method ? method : "-" ) << "]"
        << "\n"
        ;

    return spec_ff ;
}

template NPFold* NPFold::subfold_summary( const char*, const char* ) const ;
template NPFold* NPFold::subfold_summary( const char*, const char*, const char* ) const ;
template NPFold* NPFold::subfold_summary( const char*, const char*, const char*, const char* ) const ;


/**
NPFold::compare_subarrays
----------------------------

1. access *key* array from two subfold (*asym* and *bsym*)
   eg A000 and B000 which could be Opticks and Geant4 events

2. look for "subcount" summary arrays in the two folders,
   "subcount" sumaries contain array counts from multiple folders


**/

template<typename F, typename T>
NP* NPFold::compare_subarrays(const char* key, const char* asym, const char* bsym,  std::ostream* out  )
{
    NPFold* af = find_subfold_(asym) ;
    NPFold* bf = find_subfold_(bsym) ;
    NP* a = af ? af->get_(key) : nullptr ;
    NP* b = bf ? bf->get_(key) : nullptr ;

    const NP* a_subcount = af ? af->get("subcount") : nullptr ;
    const NP* b_subcount = bf ? bf->get("subcount") : nullptr ;

    int a_column = -1 ;
    int b_column = -1 ;

    NP* boa = NPX::BOA<F,T>( a, b, a_column, b_column, out );

    if(out) *out
       << "[NPFold::compare_subarray"
       << " key " << key
       << " asym " << asym
       << " bsym " << bsym
       << " af " << ( af ? "YES" : "NO " )
       << " bf " << ( bf ? "YES" : "NO " )
       << " a " << ( a ? "YES" : "NO " )
       << " b " << ( b ? "YES" : "NO " )
       << " a_subcount " << ( a_subcount ? "YES" : "NO " )
       << " b_subcount " << ( b_subcount ? "YES" : "NO " )
       << " boa " << ( boa ? "YES" : "NO " )
       << "\n"
       << "-[NPFold::compare_subarray.a_subcount" << "\n"
       << ( a_subcount ? a_subcount->descTable<int>(8) : "-\n" )
       << "-]NPFold::compare_subarray.a_subcount"
       << "\n"
       << "-[NPFold::compare_subarray.b_subcount" << "\n"
       << ( b_subcount ? b_subcount->descTable<int>(8) : "-\n" )
       << "-]NPFold::compare_subarray.b_subcount"
       << "\n"
       << "-[NPFold::compare_subarray." << asym << "\n"
       << ( a ? a->descTable<T>(8) : "-\n" )
       << "-]NPFold::compare_subarray." << asym
       << "\n"
       << "-[NPFold::compare_subarray." << bsym << "\n"
       << ( b ? b->descTable<T>(8) : "-\n" )
       << "-]NPFold::compare_subarray." << bsym
       << "\n"
       << "-[NPFold::compare_subarray.boa " << "\n"
       << ( boa ? boa->descTable<F>(12) : "-\n" )
       << "-]NPFold::compare_subarray.boa "
       << "\n"
       << "]NPFold::compare_subarray"
       << "\n"
       ;
    return boa ;
}

template<typename F, typename T>
std::string NPFold::compare_subarrays_report(const char* key, const char* asym, const char* bsym )
{
    std::stringstream ss ;
    compare_subarrays<F, T>(key, asym, bsym, &ss );
    std::string str = ss.str();
    return str ;
}



/**
NPFold::Subkey
------------------

Collect union of all keys from all the subs that are present in the metadata
of all the subfold.

**/

inline void NPFold::Subkey(std::vector<std::string>& ukey, const std::vector<const NPFold*>& subs ) // static
{
    int num_sub = subs.size();
    for(int i=0 ; i < num_sub ; i++)
    {
        std::vector<std::string> keys ;
        bool only_with_profiles = false ;
        subs[i]->getMetaKV(&keys, nullptr, only_with_profiles );
        int num_keys = keys.size();
        for(int j=0 ; j < num_keys ; j++)
        {
            const char* k = keys[j].c_str();
            if(std::find(ukey.begin(), ukey.end(), k ) == ukey.end()) ukey.push_back(k) ;
        }
    }
}

/**
NPFold::SubCommonKV
---------------------

Return (k,v) pairs that are in common for all the subs

1. collect union of all keys present in the metadata of all the subfold
2. for each of the union of keys iterate over all the subs and add entries
   with the common values into *ckey*, *cval*.
   Other keys with varying values are added to *okey*

**/

inline void NPFold::SubCommonKV(std::vector<std::string>& okey, std::vector<std::string>& ckey, std::vector<std::string>& cval, const std::vector<const NPFold*>& subs ) // static
{
    // 1. collect union of all keys present in the metadata of all the subfold
    std::vector<std::string> ukey ;
    Subkey( ukey, subs );

    int num_sub = subs.size();
    int num_ukey = ukey.size();

    bool dump_ukey = false ;
    if(dump_ukey)
    {
        std::cout << "[NPFold::SubCommonKV num_ukey:" << num_ukey ;
        for(int i=0 ; i < num_ukey ; i++ ) std::cout << ukey[i] << "\n" ;
        std::cout << "]NPFold::SubCommonKV num_ukey:" << num_ukey ;
    }

    ckey.clear();
    cval.clear();

    for(int i=0 ; i < num_ukey ; i++)
    {
        const char* k = ukey[i].c_str();

        int total = 0 ;
        int same = 0 ;
        std::string v0 ;

        for(int j=0 ; j < num_sub ; j++)
        {
            std::string v = subs[j]->get_meta_string(k) ;
            bool has_key = !v.empty();
            if(!has_key) std::cerr
                 << "NPFold::SubCommonKV MISSING KEY "
                 << " num_sub " << num_sub
                 << " num_ukey " << num_ukey
                 << " k " << ( k ? k : "-" )
                 << " v " << ( v.empty() ? "-" : v )
                 << std::endl
                 ;
            if(!has_key) std::raise(SIGINT) ;
            assert(has_key);

            total += 1 ;
            if(v0.empty())
            {
                v0 = v ;
                same += 1 ;
            }
            else
            {
                bool same_value = strcmp(v0.c_str(), v.c_str())==0 ;
                if(same_value) same += 1 ;
            }
        }

        bool all_same_value = total == same ;
        if(all_same_value)
        {
            ckey.push_back(k);
            cval.push_back(v0);
        }
        else
        {
            okey.push_back(k);
        }
    }
}

inline std::string NPFold::DescCommonKV(
     const std::vector<std::string>& okey,
     const std::vector<std::string>& ckey,
     const std::vector<std::string>& cval ) // static
{
    assert( ckey.size() == cval.size() );
    int num_ckey = ckey.size();
    int num_okey = okey.size();
    std::stringstream ss ;
    ss
       << "[NPFold::DescCommonKV" << std::endl
       << "-num_ckey " << num_ckey << std::endl
       ;
    for(int i=0 ; i < num_ckey ; i++) ss
         << std::setw(25) << ckey[i]
         << " : "
         << std::setw(25) << cval[i]
         << std::endl
         ;

    ss << "-num_okey "  << num_okey
       << std::endl
       ;
    for(int i=0 ; i < num_okey ; i++) ss
         << std::setw(25) << okey[i]
         << std::endl
         ;

    ss
       << "]NPFold::DescCommonKV"
       << std::endl
       ;

    std::string str = ss.str();
    return str ;
}



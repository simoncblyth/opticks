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
#include <cstdio>
#include <sys/types.h>

#ifdef WITH_FTS
#include <fts.h>
#endif

#include <cstring>
#include <errno.h>
#include <sstream>
#include <iomanip>

#include "NP.hh"
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

    static constexpr const int UNDEF = -1 ; 
    static constexpr const bool VERBOSE = false ; 
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
private:
    void check_integrity() const ; 
public:

    // [subfold handling 
    void         add_subfold(const char* f, NPFold* fo ); 
    int          get_num_subfold() const ;
    NPFold*      get_subfold(unsigned idx) const ; 
    const char*  get_subfold_key(unsigned idx) const ; 
    int          get_subfold_idx(const char* f) const ; 
    NPFold*      get_subfold(const char* f) const ; 
    bool         has_subfold(const char* f) const ; 


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




    static int Traverse_r(const NPFold* nd, std::string nd_path, 
          std::vector<const NPFold*>& folds, 
          std::vector<std::string>& paths ); 
    static std::string Concat(const char* base, const char* sub, char delim='/' ); 

    std::string desc_subfold(const char* top=TOP) const ;  
    void find_subfold_with_prefix(
         std::vector<const NPFold*>& subs, 
         std::vector<std::string>* subpaths,
         const char* prefix ) const ; 

    bool is_empty() const ; 
    int total_items() const ; 
    // ]subfold handling 


    void add( const char* k, const NP* a); 
    void add_(const char* k, const NP* a); 
    void set( const char* k, const NP* a); 

    static void SplitKeys( std::vector<std::string>& elem , const char* keylist, char delim=','); 
    static std::string DescKeys( const std::vector<std::string>& elem, char delim=',' ); 

    void clear(); 
private:
    void clear_(const std::vector<std::string>* keep); 
public:
    void clear_except(const char* keylist=nullptr, bool copy=true, char delim=','); 


    NPFold* copy( const char* keylist, bool shallow, char delim=',' ) const ; 
    NPFold* copy_all(bool shallow) const ; 
    static void CopyMeta( NPFold* b , const NPFold* a ); 


    int count_keys( const std::vector<std::string>* keys ) const ; 


    // single level (non recursive) accessors

    int num_items() const ; 
    const char* get_key(unsigned idx) const ; 
    const NP*   get_array(unsigned idx) const ; 

    int find(const char* k) const ; 
    bool has_key(const char* k) const ; 
    bool has_all_keys(const char* keys, char delim=',') const ; 


    const NP* get(const char* k) const ; 
    NP*       get_(const char* k); 


    const NP* get_optional(const char* k) const ; 
    int   get_num(const char* k) const ;   // number of items in array 
    void  get_counts( std::vector<std::string>* keys, std::vector<int>* counts ) const ; 
    static std::string DescCounts(const std::vector<std::string>& keys, const std::vector<int>& counts ); 




    template<typename T> T    get_meta(const char* key, T fallback=0) const ;  // for T=std::string must set fallback to ""
    std::string get_meta_string(const char* key) const ;  // empty when not found

    template<typename T> void set_meta(const char* key, T value ) ;  
 

    void save(const char* base, const char* rel) ; 
    void save(const char* base) ; 
    void save_verbose(const char* base) ; 

    void _save(const char* base) ; 
    void _save_arrays(const char* base); 
    void _save_subfold_r(const char* base); 

    void load_array(const char* base, const char* relp); 
    void load_subfold(const char* base, const char* relp);

#ifdef WITH_FTS
    static int FTS_Compare(const FTSENT** one, const FTSENT** two); 
    int  no_longer_used_load_fts(const char* base) ; 
#endif

    int  load_dir(const char* base) ; 
    int  load_index(const char* base) ; 

    int load(const char* base ) ; 
    int load(const char* base, const char* rel0, const char* rel1=nullptr ) ; 


    std::string descKeys() const ; 
    std::string desc() const ; 
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

    // TIMESTAMP/PROFILE COMPARISON USING SUBFOLD METADATA

    NPFold* substamp(  const char* prefix, const char* keyname) const ; 
    NPFold* subprofile(const char* prefix, const char* keyname) const ; 

    template<typename ... Args>
    NPFold* subfold_summary(const char* method, Args ... args_  ) const  ; 

    template<typename F, typename T>
    NP* compare_subarrays(const char* key, const char* asym="a", const char* bsym="b", std::ostream* out=nullptr  ); 

    template<typename F, typename T>
    std::string compare_subarrays_report(const char* key, const char* asym="a", const char* bsym="b" ); 
 

    static void Subkey(std::vector<std::string>& ukey, const std::vector<const NPFold*>& subs ); 
    static void SubCommonKV(std::vector<std::string>& ckey, std::vector<std::string>& cval, const std::vector<const NPFold*>& subs ); 
    static std::string DescCommonKV(const std::vector<std::string>& ckey, const std::vector<std::string>& cval ); 

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


    std::string s = ss.str(); 
    return s ; 
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
    verbose_(VERBOSE)
{
    if(verbose_) std::cerr << "NPFold::NPFold" << std::endl ; 
}

inline void NPFold::set_verbose( bool v )
{
    verbose_ = v ;  
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
inline void NPFold::add_subfold(const char* f, NPFold* fo )
{
    if(fo == nullptr) return ; 
    ff.push_back(f); // subfold keys 
    subfold.push_back(fo); 
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



inline const char* NPFold::get_subfold_key(unsigned idx) const 
{
    return idx < ff.size() ? ff[idx].c_str() : nullptr ; 
}
inline int NPFold::get_subfold_idx(const char* f) const
{
    size_t idx = std::distance( ff.begin(), std::find( ff.begin(), ff.end(), f )) ; 
    return idx < ff.size() ? idx : UNDEF ; 
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
    Traverse_r( this, "",  folds, paths ); 
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
NPFold::Traverse_r
-------------------

Traverse starting from a single NPFold and proceeding 
recursively to visit its subfold and their subfold and so on. 
Collects all folds and paths in vectors, where the paths 
are concatenated from the keys at each recursion level just 
like a file system.

**/

inline int NPFold::Traverse_r(const NPFold* nd, std::string path, 
                 std::vector<const NPFold*>& folds, std::vector<std::string>& paths ) // static
{
    folds.push_back(nd); 
    paths.push_back(path); 

    assert( nd->subfold.size() == nd->ff.size() ); 
    unsigned num_sub = nd->subfold.size(); 

    int tot_items = nd->num_items() ; 

    for(unsigned i=0 ; i < num_sub ; i++) 
    {
        const NPFold* sub = nd->subfold[i] ; 
        std::string subpath = Concat(path.c_str(), nd->ff[i].c_str(), '/' ) ;  

        int num = Traverse_r( sub, subpath,  folds, paths );  
        tot_items += num ; 
    }
    return tot_items ; 
}

inline std::string NPFold::Concat(const char* base, const char* sub, char delim ) // static
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

    int tot_items = Traverse_r( this, top,  folds, paths ); 

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

inline void NPFold::find_subfold_with_prefix(
    std::vector<const NPFold*>& subs, 
    std::vector<std::string>* subpaths, 
    const char* prefix ) const 
{
    std::vector<const NPFold*> folds ;
    std::vector<std::string>   paths ;


    int tot_items = Traverse_r( this, TOP,  folds, paths ); 

    assert( folds.size() == paths.size() ); 
    int num_paths = paths.size(); 

    bool dump = false ; 

    if(dump)
    {
        std::cerr 
            << "NPFold::find_subfold_with_prefix"
            << " prefix " << ( prefix ? prefix : "-" )
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
    
    int tot_items = Traverse_r( this, TOP,  folds, paths ); 
    return tot_items ; 
}


// ] subfold handling 


/**
NPFold::add
------------

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

**/
inline void NPFold::add_(const char* k, const NP* a) 
{
    if(verbose_) std::cerr << "NPFold::add_ [" << k  << "]" <<  std::endl ; 

    bool have_key_already = std::find( kk.begin(), kk.end(), k ) != kk.end() ; 
    if(have_key_already) std::cerr 
        << "NPFold::add_ FATAL : have_key_already [" << k << "]"  
        << std::endl 
        << descKeys()
        ; 
    assert( !have_key_already ); 

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
    if(verbose_) std::cerr << "NPFold::clear ALL" << std::endl ; 
    clear_(nullptr);     
}


/**
NPFold::clear_
----------------

This method is private as it must be used in conjunction with 
NPFold::clear_except in order to to keep (key, array) pairs
of listed keys. 

1. check_integrity (non-recursive)
2. each NP array with corresponding key not in the keep list is deleted 
3. clears the kk and aa vectors
4. for each subfold call NPFold::clear on it and clear the subfold and ff vectors

**/

inline void NPFold::clear_(const std::vector<std::string>* keep)
{
    check_integrity(); 

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        const std::string& k = kk[i] ; 
        bool listed = keep && std::find( keep->begin(), keep->end(), k ) != keep->end() ; 
        if(!listed) delete a ; 
    } 
    aa.clear(); 
    kk.clear();  

    // HUH: CLEARS ARRAY POINTER VECTOR BUT DOES NOT DELETE 
    // ARRAYS WITH KEYS IN THE KEEP LIST SO IT LOOSES 
    // ARRAY POINTERS OF KEPT ARRAYS  
    //
    // THAT CAN ONLY WORK IF THOSE POINTERS WERE GRABBED PREVIOUSLY 
    // AS THEY ARE BY clear_except

    for(unsigned i=0 ; i < subfold.size() ; i++)
    {
        NPFold* sub = const_cast<NPFold*>(subfold[i]) ; 
        sub->clear();  
    }

    subfold.clear();
    ff.clear(); 
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

**/

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

    check_integrity(); 

    std::vector<std::string> keep ; 
    if(keeplist) SplitKeys(keep, keeplist, delim); 

    std::vector<const NP*>   tmp_aa ; 
    std::vector<std::string> tmp_kk ; 

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        const std::string& k = kk[i] ; 
        bool listed = keeplist && std::find( keep.begin(), keep.end(), k ) != keep.end() ; 
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


CURRENTLY subfold are not copied. 

**/

inline NPFold* NPFold::copy( const char* keylist, bool shallow, char delim ) const 
{
    check_integrity(); 

    std::vector<std::string> keys ; 
    if(keylist) SplitKeys(keys, keylist, delim); 
    // SplitKeys adds .npy to keys if not already present 

    int count = count_keys(&keys) ; 
    if( count == 0 ) std::cerr
        << "NPFold::copy"
        << " NOTE COUNT_KEYS GIVING ZERO "
        << " keylist [" << ( keylist ? keylist : "-" ) << "]" 
        << " keylist.keys [" << DescKeys(keys, delim) << "]"  
        << " count " << count 
        << " kk.size " << kk.size() 
        << " DescKeys(kk) [" << DescKeys(kk,',')  << "]" 
        << " meta " << ( meta.empty() ? "EMPTY" : meta )  
        << std::endl 
        ; 

    //if( count == 0 ) return nullptr ; 
    // sometimes want fold metadata without any arrays

    NPFold* f = new NPFold ; 
    CopyMeta(f, this);  // copy metadata to the new fold

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        const char* k = kk[i].c_str() ; 
        bool listed = keylist && std::find( keys.begin(), keys.end(), k ) != keys.end() ; 
        if(listed)
        { 
            f->add_( k, shallow ? a : NP::MakeCopy(a) ); 
        }
    } 
    return f ; 
}

inline NPFold* NPFold::copy_all(bool shallow) const 
{
    check_integrity(); 

    NPFold* f = new NPFold ; 
    CopyMeta(f, this);  // copy metadata to the new fold

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        const char* k = kk[i].c_str() ; 
        f->add_( k, shallow ? a : NP::MakeCopy(a) ); 
    } 
    return f ; 
}







inline void NPFold::CopyMeta( NPFold* b , const NPFold* a ) // static
{
    b->meta = a->meta ; 
    b->names = a->names ; 
    b->savedir = a->savedir ? strdup(a->savedir) : nullptr ; 
    b->loaddir = a->loaddir ? strdup(a->loaddir) : nullptr ; 
    b->nodata  = a->nodata ; 
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
    int num_q = qq.size() ; 

    int q_count = 0 ; 
    for(int i=0 ; i < num_q ; i++) 
    {
       const char* q = qq[i].c_str() ; 
       if(has_key(q)) q_count += 1 ; 
    }
    bool has_all = num_q > 0 && q_count == num_q ; 
    return has_all ; 
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
    return NP::get_meta_string(meta, key);  
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






inline void NPFold::save(const char* base_, const char* rel) // not const as sets savedir
{
    std::string _base = U::form_path(base_, rel); 
    const char* base = _base.c_str(); 
    save(base); 
}


/**
NPFold::save
--------------

ISSUE : repeated use of save for a fold with no .npy ie with only subfolds
never truncates the index, so it just keeps growing at every save. 

FIXED THIS BY NOT EARLY EXITING NP::WriteNames when kk.size is zero
SO THE INDEX ALWAYS GETS TRUNCATED

**/

inline void NPFold::save(const char* base_)  // not const as calls _save
{
    const char* base = U::Resolve(base_); 

    if(base == nullptr) std::cerr 
        << "NPFold::save(\"" << ( base_ ? base_ : "-" ) << "\")"
        << " did not resolve all tokens in argument "
        << std::endl
        ;
    if(base == nullptr) return ; 

    _save(base) ; 
}

inline void NPFold::save_verbose(const char* base_)  // not const as calls _save
{
    const char* base = U::Resolve(base_); 
    std::cerr 
        << "NPFold::save(\"" << ( base_ ? base_ : "-" ) << "\")" 
        << std::endl 
        << " resolved to  [" << ( base ? base : "ERR-FAILED-TO-RESOLVE-TOKENS" ) << "]" 
        << std::endl
        ;
    if(base == nullptr) return ; 
    _save(base) ; 
}


inline void NPFold::_save(const char* base)  // not const as sets savedir
{
    assert( !nodata ); 
    savedir = strdup(base); 

    NP::WriteNames(base, INDEX, kk );  

    _save_arrays(base); 

    NP::WriteNames(base, INDEX, ff, 0, true  ); // append:true : write subfold keys (without .npy ext) to INDEX  

    _save_subfold_r(base); 

    if(!meta.empty()) U::WriteString(base, META, meta.c_str() );  

    NP::WriteNames_Simple(base, NAMES, names) ; 
}




inline void NPFold::_save_arrays(const char* base) // using the keys with .npy ext as filenames
{
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
        }
    }
    // this motivated adding directory creation to NP::save 
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
    std::vector<std::string> names ; 
    U::DirList(names, base) ; 
    if(names.size() == 0) return 1 ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str(); 
        int type = U::PathType(base, name) ; 

        if( type == U::FILE_PATH && U::EndsWith(name, "_meta.txt"))
        {
            if(VERBOSE) std::cerr << "NPFold::load_dir SKIP metadata sidecar " << name << std::endl ;
        }
        else if( type == U::FILE_PATH ) 
        {
            load_array(_base, name) ; 
        }
        else if( type == U::DIR_PATH && U::StartsWith(name, "_"))
        {
            if(VERBOSE) std::cerr << "NPFold::load_dir SKIP directory starting with _" << name << std::endl ;
        }
        else if( type == U::DIR_PATH ) 
        {
            load_subfold(_base, name);  // instanciates NPFold and add_subfold
        }
    }
    return 0 ; 
}


inline int NPFold::load_index(const char* _base) 
{
    const char* base = nodata ? _base + 1 : _base ;  
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
    return 0 ; 
}

/**
NPFold::load
---------------

HMM : note the structural difference when loading with index and 
with just loading from file system with fts

* with index get NPFold instances for each directory with keys without slash   
* with fts get single NPFold with path keys with slash  

So only get the same if call on leaf directories 

How to avoid this structural difference and allow booting from property text files ? 

* currently array/subfold decision is based .npy
* need another way to detect 
 

**/

inline int NPFold::load(const char* _base) 
{
    nodata = NP::IsNoData(_base) ;  // _path starting with NP::NODATA_PREFIX eg '@' 
    const char* base = nodata ? _base + 1 : _base ;  

    loaddir = strdup(base); 
    bool has_meta = NP::Exists(base, META) ; 
    if(has_meta) meta = U::ReadString( base, META ); 

    bool has_names = NP::Exists(base, NAMES) ; 
    if(has_names) NP::ReadNames( base, NAMES, names ); 

    bool has_index = NP::Exists(base, INDEX) ; 
    int rc = has_index ? load_index(_base) : load_dir(_base) ; 

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

inline void NPFold::getMetaKVS(
    std::vector<std::string>* keys, 
    std::vector<std::string>* vals, 
    std::vector<int64_t>* stamps, 
    bool only_with_stamp ) const 
{
    NP::GetMetaKVS(meta, keys, vals, stamps, only_with_stamp ); 
}

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


inline std::string NPFold::desc(int depth) const  
{
    std::stringstream ss ; 
    ss << "NPFold::desc( " << depth << ")" << std::endl ; 
    ss << brief() << std::endl ; 
    ss << descMetaKVS() << std::endl ; 
    for(unsigned i=0 ; i < kk.size() ; i++) 
    {
        const char* k = kk[i].c_str() ; 
        const NP* a = aa[i] ; 
        ss 
           << ( a && a->nodata ? "ND " : "   " )
           << Indent(depth*10) 
           << std::setw(20) << k 
           << " : " << ( a ? a->sstr() : "-" ) 
           << std::endl
           ;  
    }
    for(unsigned i=0 ; i < ff.size() ; i++) 
    {
        const char* f = ff[i].c_str()  ; 
        ss << std::endl << f << std::endl ;  

        NPFold* sf = subfold[i] ; 
        ss << sf->desc(depth+1) << std::endl ;   
    }
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

1. find subfold with prefix
2. get unique list of array keys from all subfold 
3. create 2d array with array counts for each sub 

**/

inline NP* NPFold::subcount( const char* prefix ) const 
{
    // 1. 
    std::vector<const NPFold*> subs ; 
    std::vector<std::string> subpaths ; 
    find_subfold_with_prefix(subs, &subpaths,  prefix );  
    assert( subs.size() == subpaths.size() ); 
    int num_sub = int(subs.size()) ; 

    // 2. 
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

    bool dump = getenv("NPFold__subcount_DUMP") != nullptr  ; 
    if(dump) std::cout << "[NPFold.hh:subcount" << std::endl ; 
    if(dump) std::cout <<  " num_ukey " << num_ukey << std::endl ;
    if(dump) for(int i=0 ; i < num_ukey ; i++ ) std::cout << a->names[i] << std::endl ; 

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

            if(dump) std::cout 
                << std::setw(20) << uk 
                << " idx " << idx 
                << " count " << count 
                << std::endl 
                ; 
        }
    }
    if(dump) std::cout << "]NPFold.hh:subcount" << std::endl ; 
    return a ; 
}



/**
NPFold::substamp
--------------------

1. finds vector of subfold of this fold with the path prefix, eg "//p" "//n" 
2. access metadata stamps for all the subfold, those with the same stamp keys
   as the first are collected into a summary stamps array 
3. labels array of the common stamp keys 
4. these arrays are returns in an NPFold

**/

inline NPFold* NPFold::substamp(const char* prefix, const char* keyname) const 
{ 
    std::vector<const NPFold*> subs ; 
    std::vector<std::string> subpaths ; 
    find_subfold_with_prefix(subs, &subpaths,  prefix );  
    assert( subs.size() == subpaths.size() ); 
    int num_sub = int(subs.size()) ; 
    int num_stamp0 = num_sub > 0 ? subs[0]->getMetaNumStamp() : 0 ;  
    bool skip = num_sub == 0 || num_stamp0 == 0 ; 

    bool dump = getenv("NPFold__substamp_DUMP") != nullptr ; 

    if(dump) std::cout 
        << "[NPFold::substamp" 
        << " find_subfold_with_prefix " << prefix
        << " num_sub " << num_sub
        << " num_stamp0 " << num_stamp0
        << " skip " << ( skip ? "YES" : "NO ") 
        << std::endl
        ;

    NPFold* out = nullptr ; 
    if(!skip)
    {
        int ni = num_sub ; 
        int nj = num_stamp0 ; 

        NP* t = NP::Make<int64_t>( ni, nj ) ; 
        int64_t* tt = t->values<int64_t>() ; 
        t->set_meta<std::string>("creator","NPFold::substamp"); 
        t->set_meta<std::string>("base", loaddir ? loaddir : "-" ); 
        t->set_meta<std::string>("prefix", prefix ? prefix : "-" ); 
        t->set_meta<std::string>("keyname", keyname ? keyname : "-" ); 

        // collect metadata (k,v) pairs that are the same for all the subs
        std::vector<std::string> ckey ;  
        std::vector<std::string> cval ; 
        SubCommonKV(ckey, cval, subs );
        assert( ckey.size() == cval.size() ); 
        t->setMetaKV_(ckey, cval); 


        std::vector<std::string> comkeys ; 
        for(int i=0 ; i < ni ; i++) 
        {
            const NPFold* sub = subs[i] ; 
            const char* subpath = subpaths[i].c_str() ; 
            std::vector<std::string> keys ; 
            std::vector<int64_t>   stamps ; 
            bool only_with_stamps = true ; 
            sub->getMetaKVS(&keys, nullptr, &stamps, only_with_stamps ); 
            assert( int(stamps.size()) == nj ) ; 

            if(i == 0) comkeys = keys ; 
            bool same_keys = i == 0 ? true : keys == comkeys ; 
            if(dump) std::cout << sub->loaddir << " stamps.size " << stamps.size() << " " << ( same_keys ? "Y" : "N" ) << std::endl; 
            assert(same_keys); 

            for(int j=0 ; j < nj ; j++) tt[i*nj+j] = stamps[j] ; 
            t->names.push_back(subpath); 

        }
        t->labels = new std::vector<std::string>(comkeys.begin(), comkeys.end())  ; 

        //NP* l = NPX::MakeCharArray(comkeys); 
        //l->names = comkeys ; 

        NP* dt = NP::DeltaColumn<int64_t>(t); 
        dt->names = t->names ; 
        dt->labels = new std::vector<std::string>(comkeys.begin(), comkeys.end())  ; 

        NP* count = subcount(prefix); 

        out = new NPFold ; 
        out->add(keyname, t );
        out->add(U::FormName("delta_",keyname,nullptr), dt );
        out->add("subcount", count ); 

    }
    if(dump) std::cout 
        << "]NPFold::substamp" 
        << std::endl
        ;
    return out ; 
}



/**
NPFold::subprofile
--------------------

1. finds vector of subfold of this fold with the path prefix, eg "//p" "//n" 
2. access metadata profiles for all of the subfold, those with the same stamp keys
   as the first are collected into a summary profile array

   * hence choose prefix to select sub NPFold with the same profile content 
 
3. labels array of the common stamp keys 
4. these arrays are returns in an NPFold

**/

inline NPFold* NPFold::subprofile(const char* prefix, const char* keyname) const 
{
    std::vector<const NPFold*> subs ; 
    std::vector<std::string> subpaths ; 
    find_subfold_with_prefix(subs, &subpaths,  prefix );  
    assert( subs.size() == subpaths.size() ); 

    int num_sub = int(subs.size()) ; 
    int num_prof0 = num_sub > 0 ? subs[0]->getMetaNumProfile() : 0 ;  
    bool skip = num_sub == 0 || num_prof0 == 0 ; 

    bool dump = false ; 

    if(dump) std::cout 
        << "[NPFold::subprofile"
        << " find_subfold_with_prefix " << prefix
        << " num_sub " << num_sub 
        << " num_prof0 " << num_prof0
        << " skip " << ( skip ? "YES" : "NO ") 
        << std::endl
        ; 

    NPFold* out = nullptr ; 
    if(!skip)
    {
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
        std::vector<std::string> ckey ;  
        std::vector<std::string> cval ; 
        SubCommonKV(ckey, cval, subs );
        assert( ckey.size() == cval.size() ); 
        t->setMetaKV_(ckey, cval); 

        std::vector<std::string> comkeys ; 
        for(int i=0 ; i < ni ; i++) 
        {
            const NPFold* sub = subs[i] ; 
            const char* subpath = subpaths[i].c_str() ; 

            if(dump) std::cout 
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
            if(dump) std::cout 
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

        NP* l = NPX::MakeCharArray(comkeys); 
        l->names = comkeys ; 

        out = new NPFold ; 
        out->add(keyname, t );
        out->add("labels", l ) ; 
    }
    std::cout 
        << "]NPFold::subprofile" 
        << std::endl
        ;
    return out ; 
}



/**
NPFold::subfold_summary
-----------------------

::

   NPFold* ab = NPFold::subfold_summary("substamp",   "a://p", "b://n" ) ; 
   NPFold* ab = NPFold::subfold_summary("subprofile", "a://p", "b://n" ) ; 

**/

template<typename ... Args>
inline NPFold* NPFold::subfold_summary(const char* method, Args ... args_  ) const 
{
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


    std::stringstream hh ; 
    hh << "NPFold::subfold_summary(\"" << method << "\"," ; 

    NPFold* spec_ff = nullptr ; 

    for(int i=0 ; i < num_uargs ; i++)
    {
        const std::string& arg = uargs[i] ; 
        hh << "\"" << arg << "\"" << ( i < num_uargs - 1 ? "," : " " ) ; 

        size_t pos = arg.find(delim) ; 
        std::string _k = arg.substr(0, pos);
        std::string _v = arg.substr(pos+1);
        const char* k = _k.c_str(); 
        const char* v = _v.c_str(); 

        NPFold* sub = nullptr ; 
        if(strcmp(method, "substamp")==0)
        {
            sub = substamp(v, "substamp") ; 
        }
        else if(strcmp(method, "subprofile")==0)
        {
            sub = subprofile(v, "subprofile") ; 
        } 

        if(sub == nullptr ) 
        {
            std::cerr 
                << "NPFold::subfold_summary"
                << " k [" << k << "]"
                << " v [" << v << "]"
                << " sub " << ( sub ? "YES" : "NO " ) 
                << std::endl 
                ; 

            continue ; 
        }
        if(spec_ff == nullptr) spec_ff = new NPFold ; 
        spec_ff->add_subfold(k, sub );  
    }   
    hh << ")" ;  

    if(spec_ff) spec_ff->headline = hh.str(); 
    return spec_ff ;  
}

template NPFold* NPFold::subfold_summary( const char*, const char* ) const ;
template NPFold* NPFold::subfold_summary( const char*, const char*, const char* ) const ;
template NPFold* NPFold::subfold_summary( const char*, const char*, const char*, const char* ) const ;


/**
NPFold::compare_subarrays
----------------------------

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
       << std::endl 
       << "-NPFold::compare_subarray.a_subcount" 
       << std::endl 
       << ( a_subcount ? a_subcount->descTable<int>(8) : "-" )
       << std::endl 
       << "-NPFold::compare_subarray.b_subcount" 
       << std::endl 
       << ( b_subcount ? b_subcount->descTable<int>(8) : "-" )
       << std::endl 
       << "-NPFold::compare_subarray." << asym 
       << std::endl
       << ( a ? a->descTable<T>(8) : "-" )
       << std::endl
       << "-NPFold::compare_subarray." << bsym 
       << std::endl
       << ( b ? b->descTable<T>(8) : "-" )
       << std::endl
       << "-NPFold::compare_subarray.boa "
       << std::endl 
       << ( boa ? boa->descTable<F>(12) : "-" ) 
       << std::endl
       << "]NPFold::compare_subarray"
       << std::endl
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

Collect union of all keys present in the metadata 
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

**/


inline void NPFold::SubCommonKV(std::vector<std::string>& ckey, std::vector<std::string>& cval, const std::vector<const NPFold*>& subs ) // static
{
    std::vector<std::string> ukey ; 
    Subkey( ukey, subs ); 

    int num_sub = subs.size(); 
    int num_ukey = ukey.size(); 

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
    }
}

inline std::string NPFold::DescCommonKV(const std::vector<std::string>& ckey, const std::vector<std::string>& cval ) // static
{
    assert( ckey.size() == cval.size() ); 
    int num_key = ckey.size(); 
    std::stringstream ss ; 
    ss << "NPFold::DescCommonKV" << std::endl ; 
    for(int i=0 ; i < num_key ; i++) ss 
         << std::setw(25) << ckey[i] 
         << " : " 
         << std::setw(25) << cval[i] 
         << std::endl 
         ;
    std::string str = ss.str(); 
    return str ; 
}



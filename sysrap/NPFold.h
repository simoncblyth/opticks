#pragma once
/**
NPFold.h : collecting and persisting NP arrays keyed by relative paths
========================================================================

This does not use other sysrap headers, other than NP, NPU 
as it will likely be incorporated into np/NPU once matured. 

There are two load/save modes:

1. with index txt file "NPFold_index.txt" : the default mode
   in which the ordering of the keys are preserved
 
2. without index txt file : the ordering of keys/arrays 
   is not preserved, it will be the fts filesystem traversal order 
   This mode allows all .npy from within any directory tree to 
   be loaded into the NPFold instance.   


RECENT ADDITION : supporting NPFold within NPFold recursively
------------------------------------------------------------------

For example "Materials" NPFold containing sub-NPFold for each material and at higher 
level "Properties" NPFold containing "Materials" and "Surfaces" sub-NPFold. 

A sub-NPFold of an NPFold is simply represented by a key in the 
index that does not end with ".npy" which gets stored into ff vector. 

**/

#include <string>
#include <algorithm> 
#include <iterator> 
#include <vector> 
#include <cstdlib>
#include <cstdio>
#include <sys/types.h>
#include <fts.h>
#include <cstring>
#include <errno.h>
#include <sstream>
#include <iomanip>

#include "NP.hh"

struct NPFold 
{
    static constexpr const int UNDEF = -1 ; 
    static constexpr const bool VERBOSE = false ; 
    static constexpr const char* EXT = ".npy" ; 
    static constexpr const char* INDEX = "NPFold_index.txt" ; 
    static constexpr const char* META  = "NPFold_meta.txt" ; 

    static bool IsNPY(const char* k); 
    static std::string FormKey(const char* k); 
    static NPFold* Load(const char* base); 
    static NPFold* Load(const char* base, const char* rel); 
    static int Compare(const NPFold* a, const NPFold* b ); 
    static std::string DescCompare(const NPFold* a, const NPFold* b ); 

    static int Compare(const FTSENT** one, const FTSENT** two); 
    static void Indent(int i); 

    // [subfold handling 
    void         add_subfold(const char* f, NPFold* fo ); 
    int          get_num_subfold() const ;
    NPFold*      get_subfold(unsigned idx) const ; 
    const char*  get_subfold_key(unsigned idx) const ; 
    int          get_subfold_idx(const char* f) const ; 
    NPFold*      get_subfold(const char* f) const ; 
    NPFold*      find_subfold(const char* fpath) ; 
    static std::string Concat(const char* base, const char* sub, char delim='/' ); 

    static void Traverse_r(NPFold* nd, std::string nd_path, 
          std::vector<NPFold*>& folds, 
          std::vector<std::string>& paths ); 
    std::string desc_subfold(const char* top="/");  

    std::vector<NPFold*> subfold ;  
    std::vector<std::string> ff ;  // for sub-NPFold 
    // ]subfold handling 

    std::vector<std::string> kk ; 
    std::vector<const NP*>   aa ; 
    std::string meta ; 
    const char* savedir ; 
    const char* loaddir ; 

    NPFold(); 

    void check() const ; 
    void add(const char* k, const NP* a); 
    void set(const char* k, const NP* a); 
    void clear(); 

    int num_items() const ; 
    const char* get_key(unsigned idx) const ; 
    const NP*   get_array(unsigned idx) const ; 

    int find(const char* k) const ; 
    bool has_key(const char* k) const ; 

    const NP* get(const char* k) const ; 
    int   get_num(const char* k) const ; 

    void save(const char* base, const char* rel) ; 
    void save(const char* base) ; 
    void _save_arrays(const char* base); 
    void _save_subfold(const char* base); 

    int load(const char* base) ; 
    int load(const char* base, const char* rel) ; 

    int  load_fts(const char* base) ; 
    int  load_index(const char* base) ; 
    void load_array(const char* base, const char* relp); 
    void load_subfold(const char* base, const char* relp);


    std::string desc() const ; 
    std::string brief() const ; 
}; 



inline NPFold* NPFold::Load(const char* base)
{
    NPFold* nf = new NPFold ; 
    nf->load(base); 
    return nf ;  
}

inline NPFold* NPFold::Load(const char* base, const char* rel)
{
    NPFold* nf = new NPFold ; 
    nf->load(base, rel); 
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


inline NPFold::NPFold()
    :
    savedir(nullptr),
    loaddir(nullptr)
{
}


inline void NPFold::check() const
{
    assert( kk.size() == aa.size() ); 
    assert( ff.size() == subfold.size() ); 
}

inline bool NPFold::IsNPY(const char* k) // key ends with EXT ".npy"
{
    return strlen(k) > strlen(EXT) && strcmp( k + strlen(k) - strlen(EXT), EXT ) == 0 ; 
}

inline std::string NPFold::FormKey(const char* k)
{
    std::stringstream ss ; 
    ss << k ; 
    if(!IsNPY(k)) ss << EXT ; 
    std::string s = ss.str(); 
    return s ; 
}




// [ subfold handling 
inline void NPFold::add_subfold(const char* f, NPFold* fo )
{
    ff.push_back(f); 
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
inline NPFold* NPFold::find_subfold(const char* qpath) 
{
    std::vector<NPFold*> folds ;
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

inline std::string NPFold::Concat(const char* base, const char* sub, char delim ) // static
{
    assert(sub) ; // base can be nullptr : needed for root, but sub must always be defined 
    std::stringstream ss ;
    if(base && strlen(base) > 0) ss << base << delim ; 
    ss << sub ; 
    std::string s = ss.str(); 
    return s ; 
}

inline void NPFold::Traverse_r(NPFold* nd, std::string path, 
                 std::vector<NPFold*>& folds, std::vector<std::string>& paths ) // static
{
    folds.push_back(nd); 
    paths.push_back(path); 

    assert( nd->subfold.size() == nd->ff.size() ); 
    unsigned num_sub = nd->subfold.size(); 
    for(unsigned i=0 ; i < num_sub ; i++) 
    {
        NPFold* sub = nd->subfold[i] ; 
        std::string subpath = Concat(path.c_str(), nd->ff[i].c_str(), '/' ) ;  

        Traverse_r( sub, subpath,  folds, paths );  
    }
}

inline std::string NPFold::desc_subfold(const char* top) 
{
    std::vector<NPFold*>       folds ;
    std::vector<std::string>   paths ;
    Traverse_r( this, top,  folds, paths ); 

    std::stringstream ss ; 
    ss << " folds " << folds.size() << std::endl ; 
    ss << " paths " << paths.size() << std::endl ; 
    for(unsigned i=0 ; i < paths.size() ; i++) ss << i << " [" << paths[i] << "]" << std::endl ; 

    std::string s = ss.str(); 
    return s ; 
}

// ] subfold handling 


/**
NPFold::add
------------

If added keys do not end with the EXT ".npy" then the EXT is added prior to collection. 

**/

inline void NPFold::add(const char* k, const NP* a) 
{
    std::string key = FormKey(k); 
    kk.push_back(key); 
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

inline void NPFold::clear()
{
    check(); 

    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i]; 
        delete a ; 
    } 
    aa.clear(); 
    kk.clear(); 

    for(unsigned i=0 ; i < subfold.size() ; i++)
    {
        NPFold* sub = const_cast<NPFold*>(subfold[i]) ; 
        sub->clear();  
    }

    subfold.clear();
    ff.clear(); 
}


inline int NPFold::num_items() const 
{
    check(); 
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
NPFold::find
-------------

If the key *k* does not ext with EXT ".npy" then that is added before searching.

std::find returns iterator to the first match

**/
inline int NPFold::find(const char* k) const
{
    std::string key = FormKey(k); 
    size_t idx = std::distance( kk.begin(), std::find( kk.begin(), kk.end(), key.c_str() )) ; 
    return idx < kk.size() ? idx : UNDEF ; 
}

inline bool NPFold::has_key(const char* k) const 
{
    int idx = find(k); 
    return idx != UNDEF  ; 
}

inline const NP* NPFold::get(const char* k) const 
{
    int idx = find(k) ; 
    return idx == UNDEF ? nullptr : aa[idx] ; 
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



inline void NPFold::save(const char* base_, const char* rel) // not const as sets savedir
{
    std::string base = NP::form_path(base_, rel); 
    save(base.c_str()); 
}

inline void NPFold::save(const char* base)  // not const as sets savedir
{
    savedir = strdup(base); 

    NP::WriteNames(base, INDEX, kk );  

    _save_arrays(base); 

    NP::WriteNames(base, INDEX, ff, 0, true  ); // append write subfold keys to INDEX  
    _save_subfold(base); 

    if(!meta.empty()) NP::WriteString(base, META, meta.c_str() );  
}

inline void NPFold::_save_arrays(const char* base)
{
    for(unsigned i=0 ; i < kk.size() ; i++) 
    {
        const char* k = kk[i].c_str() ; 
        const NP* a = aa[i] ; 
        if( a == nullptr )
        {
            std::cout << " base " << base << " k " << k << " ERROR MISSING ARRAY FOR KEY " << std::endl ;   
        }
        else
        { 
            a->save(base, k );  
        }
    }
    // this motivated adding directory creation to NP::save 
}



inline void NPFold::_save_subfold(const char* base)
{
    assert( subfold.size() == ff.size() ); 
    for(unsigned i=0 ; i < ff.size() ; i++) 
    {
        const char* f = ff[i].c_str() ; 
        NPFold* sf = subfold[i] ; 
        sf->save(base, f );  
    }
}




inline int NPFold::Compare(const FTSENT** one, const FTSENT** two)
{
    return (strcmp((*one)->fts_name, (*two)->fts_name));
}
inline void NPFold::Indent(int i)
{ 
    for(; i > 0; i--) printf("    ");
}

inline void NPFold::load_array(const char* base, const char* relp)
{
    if(IsNPY(relp)) add(relp,  NP::Load(base, relp) ) ; 
}

inline void NPFold::load_subfold(const char* base, const char* relp)
{
    assert(!IsNPY(relp)); 
    add_subfold(relp,  NPFold::Load(base, relp) ) ; 
}


 
inline int NPFold::load_fts(const char* base_) 
{
    char* base = const_cast<char*>(base_);  
    char* basepath[2] {base, nullptr};

    FTS* fs = fts_open(basepath,FTS_COMFOLLOW|FTS_NOCHDIR,&Compare);
    if(fs == nullptr) return 1 ; 

    FTSENT* node = nullptr ;
    while((node = fts_read(fs)) != nullptr)
    {   
        switch (node->fts_info) 
        {   
            case FTS_D :
                break;
            case FTS_F :
            case FTS_SL:
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

inline int NPFold::load_index(const char* base) 
{
    std::vector<std::string> keys ; 
    NP::ReadNames(base, INDEX, keys );  
    for(unsigned i=0 ; i < keys.size() ; i++) 
    {
        const char* key = keys[i].c_str() ; 
        if(IsNPY(key))
        {
            load_array(base, key );   // invokes ::add appending to kk and aa 
        }
        else
        {
            load_subfold(base, key); 
        }
    }
    return 0 ; 
}

inline int NPFold::load(const char* base) 
{
    loaddir = strdup(base); 
    bool has_meta = NP::Exists(base, META) ; 
    if(has_meta) meta = NP::ReadString( base, META ); 

    bool has_index = NP::Exists(base, INDEX) ; 
    return has_index ? load_index(base) : load_fts(base) ; 
}
inline int NPFold::load(const char* base_, const char* rel) 
{
    std::string base = NP::form_path(base_, rel); 
    return load(base.c_str()); 
}



inline std::string NPFold::desc() const  
{
    std::stringstream ss ; 
    ss << "NPFold::desc"  << std::endl ; 
    ss << brief() << std::endl ; 
    for(unsigned i=0 ; i < kk.size() ; i++) 
    {
        const char* k = kk[i].c_str() ; 
        const NP* a = aa[i] ; 
        ss << std::setw(40) << k << " : " << ( a ? a->sstr() : "-" ) << std::endl ;  
    }
    for(unsigned i=0 ; i < ff.size() ; i++) 
    {
        const char* f = ff[i].c_str()  ; 
        ss << std::setw(40) << f << std::endl ;  
    }
    std::string s = ss.str(); 
    return s ; 
}

inline std::string NPFold::brief() const 
{
    std::stringstream ss ; 
    if(loaddir) ss << " loaddir:" << loaddir ; 
    if(savedir) ss << " savedir:" << savedir ; 
    ss <<  " subfold " << subfold.size() ; 
    ss << " ff " << ff.size() ; 
    ss << " kk " << kk.size() ; 
    ss << " aa " << aa.size() ; 
    std::string s = ss.str(); 
    return s ; 
}


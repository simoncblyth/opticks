#pragma once
/**
sfreq.h : count occurrence frequencies of strings and sorts by frequencies
============================================================================

Canonical usage is for geometry progeny digests 

**/

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "NP.hh"

struct sfreq_matchkey 
{
    const char* query ; 
    sfreq_matchkey(const char* query_) : query(query_) {}

    bool operator()(const std::pair<std::string, unsigned>& p) const 
    { 
        return strcmp(query, p.first.c_str()) == 0 ;
    }
};

struct sfreq
{
    typedef std::pair<std::string,unsigned> SU ;   
    typedef typename std::vector<SU>        VSU  ; 
    typedef typename VSU::const_iterator    IT ;   

    VSU vsu ; 

    const char* get_key(unsigned idx) const ; 
    unsigned get_freq(unsigned idx) const ; 

    int find_index(const char* key) const ; 
    int get_freq(const char* key) const ; 
    void add(const char* key ); 

    static bool ascending_freq( const SU& a, const SU& b) ; 
    static bool descending_freq(const SU& a, const SU& b) ; 
    void sort(bool descending=true);  

    std::string desc() const ; 

    static constexpr const char* KEY = "key.npy" ; 
    static constexpr const char* VAL = "val.npy" ; 
    size_t get_maxkeylen() const ; 

    NP* make_key() const ; 
    NP* make_val() const ; 
    void import_key_val( const NP* key, const NP* val); 

    void save(const char* dir) const ; 
    void save(const char* dir, const char* reldir) const ; 

    void load(const char* dir); 
    void load(const char* dir, const char* reldir); 
};


inline const char* sfreq::get_key(unsigned idx) const
{
    assert( idx < vsu.size() ); 
    return vsu[idx].first.c_str() ; 
}
inline unsigned sfreq::get_freq(unsigned idx) const
{
    assert( idx < vsu.size() ); 
    return vsu[idx].second ; 
}

inline int sfreq::find_index(const char* key) const 
{
    sfreq_matchkey mk(key);  
    IT it = std::find_if( vsu.begin(), vsu.end(), mk ); 
    return it == vsu.end() ? -1 : std::distance( vsu.begin(), it ); 
}

inline int sfreq::get_freq(const char* key) const
{
    int idx = find_index(key); 
    return idx == -1 ? -1 : int(vsu[idx].second) ; 
}

inline void sfreq::add(const char* key)
{
    int idx = find_index(key); 
    if( idx == -1 ) vsu.push_back(SU(key, 1u)) ; 
    else vsu[idx].second += 1 ;  
}

inline bool sfreq::ascending_freq(const SU& a, const SU& b)  // static
{
    return b.second > a.second ;
}
inline bool sfreq::descending_freq(const SU& a, const SU& b) // static 
{
    return a.second > b.second ;
}
inline void sfreq::sort(bool descending) 
{
    std::sort(vsu.begin(), vsu.end(), descending ? descending_freq : ascending_freq );
}

inline std::string sfreq::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < vsu.size() ; i++)
    {
        const SU& su = vsu[i] ;  
        const std::string& k = su.first ; 
        unsigned v = su.second ; 

        ss << std::setw(5) << i 
           << " : " 
           << std::setw(32) << k.c_str()
           << " : " 
           << std::setw(5) << v
           << std::endl 
           ;  
    }
    std::string s = ss.str(); 
    return s ; 
}

inline size_t sfreq::get_maxkeylen() const 
{
    size_t mx = 0 ; 
    for(unsigned i=0 ; i < vsu.size() ; i++) mx = std::max(mx, strlen(vsu[i].first.c_str())) ; 
    return mx ;  
}

/**
In [5]: t.key.view("|S5").ravel()
Out[5]: array([b'blue', b'red', b'green'], dtype='|S5')

In [6]: t.key.shape
Out[6]: (3, 5)
**/


inline NP* sfreq::make_key() const 
{
    size_t mkl = get_maxkeylen() ; 
    NP* key = NP::Make<char>( vsu.size(), mkl ) ;
    char* kdat = key->values<char>(); 

    for(unsigned i=0 ; i < vsu.size() ; i++)
    {
        const std::pair<std::string, unsigned> su = vsu[i] ;  
        const char* k = su.first.c_str() ; 
        for(unsigned j=0 ; j < strlen(k) ; j++) kdat[i*mkl+j] = k[j] ; 
    }
    return key ; 
}

inline NP* sfreq::make_val() const 
{
    NP* val = NP::Make<unsigned>( vsu.size() ) ; 
    unsigned* vdat = val->values<unsigned>(); 

    for(unsigned i=0 ; i < vsu.size() ; i++)
    {
        const std::pair<std::string, unsigned> su = vsu[i] ;  
        vdat[i] = su.second ; 
    }
    return val ;
} 

inline void sfreq::import_key_val( const NP* key, const NP* val)
{
    unsigned mkl = key->shape[1] ; 
    const char* kdat = key->cvalues<char>(); 
    const unsigned* vdat = val->cvalues<unsigned>(); 

    assert( key->shape[0] == val->shape[0]) ; 
    unsigned num_kv = key->shape[0] ; 
 
    for(unsigned i=0 ; i < num_kv ; i++)
    {
        const char* kptr = kdat+i*mkl ; 
        std::string k(kptr, kptr+mkl) ; 
        unsigned v = vdat[i] ; 
        vsu.push_back(std::pair<std::string, unsigned>(k,v) );  
    }
}




inline void sfreq::save(const char* dir) const 
{
    const NP* key = make_key(); 
    const NP* val = make_val(); 
    key->save( dir, KEY) ; 
    val->save( dir, VAL) ; 
}

inline void sfreq::save(const char* dir, const char* reldir) const 
{
    const NP* key = make_key(); 
    const NP* val = make_val(); 
    key->save( dir, reldir, KEY) ; 
    val->save( dir, reldir, VAL) ; 
}

inline void sfreq::load(const char* dir)
{
    NP* key = NP::Load(dir, KEY); 
    NP* val = NP::Load(dir, VAL); 
    import_key_val(key, val); 
}
inline void sfreq::load(const char* dir, const char* reldir)
{
    NP* key = NP::Load(dir, reldir, KEY); 
    NP* val = NP::Load(dir, reldir, VAL); 
    import_key_val(key, val); 
}



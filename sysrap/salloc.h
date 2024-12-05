#pragma once
/**
salloc.h : debug out of memory errors on device
=================================================

This is used to debug out of memory errors on device. 

Some device allocations such as those by QU::device_alloc 
are monitored when the *QU:alloc* *salloc* instance 
has been instanciated. 

WIP: replace glm::tvec4 with struct 

::

    epsilon:opticks blyth$ opticks-fl salloc.h 
    ./CSGOptiX/CSGOptiX.cc
    ./sysrap/CMakeLists.txt
    ./sysrap/SEventConfig.cc
    ./sysrap/tests/SEventConfigTest.cc
    ./sysrap/tests/salloc_test.cc
    ./qudarap/QU.cc


**/

#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#ifdef OLD_GLM_SALLOC
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#else
struct salloc_item
{
    uint64_t size ; 
    uint64_t num_items ; 
    uint64_t sizeof_item ; 
    uint64_t spare ; 
};
#endif

#include "NP.hh"

struct salloc
{
    static constexpr const char* ALLOC = "salloc.npy" ; 
    static constexpr const char* RELDIR = "salloc" ; 

    std::string meta ; 
    std::vector<std::string> label ; 
#ifdef OLD_GLM_SALLOC
    std::vector<glm::tvec4<uint64_t>> alloc ; 
#else
    std::vector<salloc_item> alloc ; 
#endif
    void add(const char* label, uint64_t size, uint64_t num_items, uint64_t sizeof_item, uint64_t spare); 
    uint64_t get_total() const ; 

    NP* make_array() const ; 


    void save(const char* base, const char* reldir=RELDIR); 
    void import(const NP* aa);
    static salloc* Load(const char* base, const char* reldir=RELDIR); 

    std::string desc() const ; 

    template<typename T> void set_meta(const char* key, T value); 
}; 

inline void salloc::add( const char* label_, uint64_t size, uint64_t num_items, uint64_t sizeof_item, uint64_t spare )
{
    label.push_back(label_ ); 
    alloc.push_back( {size, num_items, sizeof_item, spare } ); 
}
inline uint64_t salloc::get_total() const 
{
    uint64_t tot = 0 ; 
    for(unsigned i=0 ; i < alloc.size() ; i++) 
    {
#ifdef OLD_GLM_SALLOC
       tot += alloc[i].x ; 
#else
       tot += alloc[i].size ;
#endif

    }
    return tot ; 
}



inline NP* salloc::make_array() const 
{
    NP* a = NP::Make<uint64_t>( alloc.size(), 4 ); 
    a->read2<uint64_t>( (uint64_t*)alloc.data() ); 
    a->set_names( label ); 
    if(!meta.empty()) a->meta = meta ; 
    return a ; 
}
inline void salloc::save(const char* base, const char* reldir )
{
    NP* a = make_array(); 
    a->save(base, reldir, ALLOC ); 
}

inline void salloc::import(const NP* a)
{
    assert( a ); 
    alloc.resize( a->shape[0] ); 
    memcpy( alloc.data(), a->bytes(), a->arr_bytes() ); 
    a->get_names(label); 
    if(!a->meta.empty()) meta = a->meta ; 
}

inline salloc* salloc::Load(const char* base, const char* reldir )
{
    NP* aa = NP::Load(base, reldir, ALLOC) ; 
    if( aa == nullptr ) return nullptr ; 

    //std::cout << "salloc::Load aa.desc " << aa->desc() << std::endl ; 
    salloc* a = new salloc ; 
    a->import(aa) ; 
    return a ; 
}

inline std::string salloc::desc() const 
{
    uint64_t tot = get_total() ; 
    double tot_GB = double(tot)/1e9 ; 

    std::stringstream ss ; 
    ss << "salloc::desc"
       << " alloc.size " << alloc.size()
       << " label.size " << label.size()
       << std::endl
       ;

    ss << "salloc.meta"
       << std::endl 
       << ( meta.empty() ? "-" : meta )
       << std::endl 
       ;

    const char* spacer = "     " ;  
    ss 
        << std::endl
        << spacer
        << "[" << std::setw(15) << "size"
        << " " << std::setw(11) << "num_items"
        << " " << std::setw(11) << "sizeof_item"
        << " " << std::setw(11) << "spare"
        << "]"
        << " " << std::setw(10) << "size_GB"
        << " " << std::setw(10) << "percent"  
        << " " << "label"
        << std::endl
        << spacer
        << "[" << std::setw(15) << "(bytes)"
        << " " << std::setw(11) << ""
        << " " << std::setw(11) << ""
        << " " << std::setw(11) << ""
        << "]"
        << " " << std::setw(10) << "size/1e9"
        << " " << std::setw(10) << ""  
        << " " << ""
        << std::endl
        << std::endl
        ;

    assert( alloc.size() == label.size() ); 
    for(unsigned i=0 ; i < alloc.size() ; i++ )
    {
        const char* lab = label[i].c_str() ; 
#ifdef OLD_GLM_SALLOC
        const glm::tvec4<uint64_t>& item = alloc[i] ;   
        uint64_t _size = item.x ;  
        uint64_t _num_items = item.y ; 
        uint64_t _sizeof_item = item.z ; 
        uint64_t _spare = item.w ; 
#else
        const salloc_item& item = alloc[i] ;   
        uint64_t _size = item.size ;  
        uint64_t _num_items = item.num_items ; 
        uint64_t _sizeof_item = item.sizeof_item ; 
        uint64_t _spare = item.spare ; 
#endif
        double size = double(_size) ; 
        double size_GB  = size/1e9 ; 
        float size_percent = 100.*size/double(tot) ; 

        ss 
            << spacer
            << "[" << std::setw(15) << _size
            << " " << std::setw(11) << _num_items
            << " " << std::setw(11) << _sizeof_item
            << " " << std::setw(11) << _spare
            << "]"
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << size_GB
            << " " << std::setw(10) << std::fixed << std::setprecision(2) << size_percent
            << " " << lab 
            << std::endl 
            ; 
    }
    ss << std::endl ; 
    ss << " tot  " << std::setw(15) << tot 
       << " " << std::setw(11) << "" 
       << " " << std::setw(11) << "" 
       << " " << std::setw(11) << "" 
       << " "
       << " " << std::setw(10) << std::fixed << std::setprecision(2) << tot_GB
       << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}

template<typename T> 
inline void salloc::set_meta(const char* key, T value)
{
    NP::SetMeta(meta, key, value ); 
}

template void salloc::set_meta<uint64_t>(const char*, uint64_t );


#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "NP.hh"

struct salloc
{
    static constexpr const char* ALLOC = "salloc.npy" ; 
    static constexpr const char* RELDIR = "salloc" ; 
    std::vector<std::string> label ; 
    std::vector<glm::tvec4<uint64_t>> alloc ; 
    void add(const char* label, uint64_t size, uint64_t num_items, uint64_t sizeof_item, uint64_t spare); 
    uint64_t get_total() const ; 

    NP* make_array() const ; 

    void save(const char* base, const char* reldir=RELDIR); 
    void load(const char* base, const char* reldir=RELDIR);

    std::string desc() const ; 
}; 

inline void salloc::add( const char* label_, uint64_t size, uint64_t num_items, uint64_t sizeof_item, uint64_t spare )
{
    label.push_back(label_ ); 
    alloc.push_back( {size, num_items, sizeof_item, spare } ); 
}
inline uint64_t salloc::get_total() const 
{
    uint64_t tot = 0 ; 
    for(unsigned i=0 ; i < alloc.size() ; i++) tot += alloc[i].x ; 
    return tot ; 
}



inline NP* salloc::make_array() const 
{
    NP* a = NP::Make<uint64_t>( alloc.size(), 4 ); 
    a->read2<uint64_t>( (uint64_t*)alloc.data() ); 
    a->set_names( label ); 
    return a ; 
}
inline void salloc::save(const char* base, const char* reldir )
{
    NP* a = make_array(); 
    a->save(base, reldir, ALLOC ); 
}
inline void salloc::load(const char* base, const char* reldir )
{
    NP* a = NP::Load(base, reldir, ALLOC) ; 
    //std::cout << "salloc::load a.desc " << a->desc() << std::endl ; 

    alloc.resize( a->shape[0] ); 
    memcpy( alloc.data(), a->bytes(), a->arr_bytes() ); 
    a->get_names(label); 
}

inline std::string salloc::desc() const 
{
    uint64_t tot = get_total() ; 
    std::stringstream ss ; 
    ss << "salloc::desc"
       << " alloc.size " << alloc.size()
       << " label.size " << label.size()
       << std::endl
       ;

    assert( alloc.size() == label.size() ); 
    for(unsigned i=0 ; i < alloc.size() ; i++ )
    {
        const char* lab = label[i].c_str() ; 
        const glm::tvec4<uint64_t>& vec = alloc[i] ;   
        ss 
            << "[" << std::setw(10) << vec.x 
            << " " << std::setw(10) << vec.y
            << " " << std::setw(10) << vec.z 
            << " " << std::setw(10) << vec.w
            << "]"
            << " " << lab 
            << std::endl 
            ; 
    }
    ss << " tot " << std::setw(10) << tot << std::endl ;;
    std::string s = ss.str(); 
    return s ; 
}



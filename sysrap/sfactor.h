#pragma once
/**
sfactor.h
===========

Note that the remainder volumes are not included in the sfactors 

::

    stree::desc_factor
    sfactor::Desc num_factor 9
    sfactor index   0 freq  25600 sensors      0 subtree      5 sub [1af760275cafe9ea890bfa01b0acb1d1]
    sfactor index   1 freq  12615 sensors      0 subtree      7 sub [0077df3ebff8aeec56c8a21518e3c887]
    sfactor index   2 freq   4997 sensors      0 subtree      7 sub [1e410142530e54d54db8aaaccb63b834]
    sfactor index   3 freq   2400 sensors      0 subtree      6 sub [019f9eccb5cf94cce23ff7501c807475]
    sfactor index   4 freq    590 sensors      0 subtree      1 sub [c051c1bb98b71ccb15b0cf9c67d143ee]
    sfactor index   5 freq    590 sensors      0 subtree      1 sub [5e01938acb3e0df0543697fc023bffb1]
    sfactor index   6 freq    590 sensors      0 subtree      1 sub [cdc824bf721df654130ed7447fb878ac]
    sfactor index   7 freq    590 sensors      0 subtree      1 sub [3fd85f9ee7ca8882c8caa747d0eef0b3]
    sfactor index   8 freq    504 sensors      0 subtree    130 sub [7d9a644fae10bdc1899c0765077e7a33]


    In [6]: st.f.factor[:,:4]
    Out[6]: 
    array([[    0, 25600, 25600,     5],
           [    1, 12615, 12615,    11],
           [    2,  4997,  4997,    14],
           [    3,  2400,  2400,     6],
           [    4,   590,     0,     1],
           [    5,   590,     0,     1],
           [    6,   590,     0,     1],
           [    7,   590,     0,     1],
           [    8,   504,   504,   130]], dtype=int32)

             index  freq  sensors  subtree
                          ???


    In [13]: f.factor[:,4:].copy().view("|S32")
    Out[13]: 
    array([[b'1af760275cafe9ea890bfa01b0acb1d1'],
           [b'0077df3ebff8aeec56c8a21518e3c887'],
           [b'1e410142530e54d54db8aaaccb63b834'],
           [b'019f9eccb5cf94cce23ff7501c807475'],
           [b'c051c1bb98b71ccb15b0cf9c67d143ee'],
           [b'5e01938acb3e0df0543697fc023bffb1'],
           [b'cdc824bf721df654130ed7447fb878ac'],
           [b'3fd85f9ee7ca8882c8caa747d0eef0b3'],
           [b'7d9a644fae10bdc1899c0765077e7a33']], dtype='|S32')

**/

#include <vector>
#include <cassert>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>


struct sfactor
{
    static constexpr const int NV = 12 ; // sub is equivalent to 8 integer fields 
    int      index ; 
    int      freq ; 
    int      sensors ; 
    int      subtree ;  // counts of nodes in subtree 
    char     sub[32] ;  // caution : no null termination (size of 8 ints)

    void        set_sub(const char* s); 
    std::string get_sub() const ; 

    std::string desc() const ; 
    static std::string Desc(const std::vector<sfactor>& factor); 

}; 

inline void sfactor::set_sub(const char* s)
{
    assert( strlen(s) == 32 ); 
    memcpy( &sub, s, 32 ); 
}

inline std::string sfactor::get_sub() const 
{
    std::string sub_(sub, 32);  // needed as sub array is not null terminated 
    return sub_ ; 
}

inline std::string sfactor::desc() const 
{
    std::stringstream ss ; 
    ss << "sfactor"
       << " index " << std::setw(3) << index
       << " freq " << std::setw(6) << freq
       << " sensors " << std::setw(6) << sensors
       << " subtree " << std::setw(6) << subtree 
       << " freq*subtree " << std::setw(7) <<  freq*subtree
       << " sub [" << std::setw(32) << get_sub()  << "]" 
       ;   
    std::string s = ss.str(); 
    return s ; 
}

inline std::string sfactor::Desc(const std::vector<sfactor>& factor) // static
{
    int num_factor = factor.size(); 
    std::stringstream ss ; 
    ss << "sfactor::Desc num_factor " << num_factor << std::endl ; 
    int tot_freq_subtree = 0 ; 
    for(int i=0 ; i < num_factor ; i++) 
    {   
        const sfactor& sf = factor[i]; 
        tot_freq_subtree += sf.freq*sf.subtree ; 
        ss << sf.desc() << std::endl ; 
    }
    ss << " tot_freq_subtree " << std::setw(7) << tot_freq_subtree << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


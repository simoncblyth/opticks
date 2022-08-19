#pragma once

#include <string>
#include <sstream>
#include <iomanip>

struct snode
{
    static constexpr const int NV = 12 ; 

    int index ; 
    int depth ; 
    int sibdex ;  // 0-based sibling index 
    int parent ; 

    int num_child ; 
    int first_child ; 
    int next_sibling ; 
    int lvid ;

    int copyno ; 
    int sensor_id ; 
    int sensor_index ; 
    int factor_index ; 

    // material index ... 

    std::string desc() const ; 
}; 


inline std::string snode::desc() const
{
    std::stringstream ss ;
    ss << "snode"
       << " ix:" << std::setw(7) << index
       << " dh:" << std::setw(2) << depth
       << " sx:" << std::setw(5) << sibdex
       << " pt:" << std::setw(7) << parent
       << " nc:" << std::setw(5) << num_child
       << " fc:" << std::setw(7) << first_child
       << " ns:" << std::setw(7) << next_sibling
       << " lv:" << std::setw(3) << lvid
       << " cp:" << std::setw(7) << copyno
       << " se:" << std::setw(7) << sensor_id
       << " se:" << std::setw(7) << sensor_index
       << " fi:" << std::setw(2) << factor_index
       ;
    std::string s = ss.str();
    return s ;
}



#pragma once

#include <string>
#include <sstream>
#include <iomanip>

struct snode
{
    static constexpr const int NV = 8 ; 

    int index ; 
    int depth ; 
    int sibdex ; 
    int parent ; 

    int num_child ; 
    int first_child ; 
    int next_sibling ; 
    int lvid ;

    std::string desc() const ; 
}; 


inline std::string snode::desc() const
{
    std::stringstream ss ;
    ss << "snode"
       << " id:" << std::setw(7) << index
       << " de:" << std::setw(2) << depth
       << " si:" << std::setw(5) << sibdex
       << " pa:" << std::setw(7) << parent
       << " nc:" << std::setw(5) << num_child
       << " fc:" << std::setw(7) << first_child
       << " ns:" << std::setw(7) << next_sibling
       << " lv:" << std::setw(3) << lvid
       ;
    std::string s = ss.str();
    return s ;
}



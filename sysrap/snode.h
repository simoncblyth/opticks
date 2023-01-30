#pragma once
/**
snode.h : structural "volume" nodes
=====================================

snode are structural nodes residing in stree.h and populated 
from Geant4 volumes by U4Tree.h U4Tree::initNodes_r

For traversal examples see::

   stree::get_children

**/

#include <string>
#include <sstream>
#include <iomanip>

struct snode
{
    static constexpr const int NV = 14 ; 

    int index ;        // 0 
    int depth ;        // 1
    int sibdex ;       // 2    0-based sibling index 
    int parent ;       // 3

    int num_child ;    // 4
    int first_child ;  // 5 
    int next_sibling ; // 6  
    int lvid ;         // 7

    int copyno ;       // 8 
    int sensor_id ;    // 9 
    int sensor_index ; // 10  
    int repeat_index ; // 11

    int repeat_ordinal ; // 12
    int boundary ;       // 13

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
       << " ri:" << std::setw(2) << repeat_index
       << " ro:" << std::setw(5) << repeat_ordinal
       << " bd:" << std::setw(2) << boundary
       ;
    std::string s = ss.str();
    return s ;
}



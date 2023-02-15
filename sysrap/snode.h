#pragma once
/**
snode.h : structural "volume" nodes
=====================================

snode are structural nodes residing in stree.h and populated 
from Geant4 volumes by U4Tree.h U4Tree::initNodes_r

There are no transform references in snode.h as all 
the below vectors are populated together so the node
index also corresponds to the other indices::

   stree::nds
   stree::digs
   stree::m2w
   stree::w2m
   stree::gtd

For traversal examples see *stree::get_children*

**/

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

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
    static std::string Brief_(const std::vector<snode>& nodes ); 


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
    std::string str = ss.str();
    return str ;
}

inline std::string snode::Brief_(const std::vector<snode>& nodes )
{
    int num_nodes = nodes.size(); 
    std::stringstream ss ;
    ss << "snode::Brief_ num_nodes " << num_nodes << std::endl ; 
    for(int i=0; i < num_nodes ; i++) ss << nodes[i].desc() << std::endl ; 
    std::string str = ss.str();
    return str ;
}

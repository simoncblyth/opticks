#pragma once
/**
snode.h : structural "volume" nodes
=====================================

snode are structural nodes residing in the the stree.h
vectors *stree::nds* and *stree::rem* that are populated
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
#include <cstring>
#include <sstream>
#include <iomanip>
#include <vector>

struct snode_field
{
    static constexpr const char* INDEX          = "INDEX" ;
    static constexpr const char* DEPTH          = "DEPTH" ;
    static constexpr const char* SIBDEX         = "SIBDEX" ;
    static constexpr const char* PARENT         = "PARENT" ;
    static constexpr const char* NUM_CHILD      = "NUM_CHILD" ;
    static constexpr const char* FIRST_CHILD    = "FIRST_CHILD" ;
    static constexpr const char* NEXT_SIBLING   = "NEXT_SIBLING" ;
    static constexpr const char* LVID           = "LVID" ;
    static constexpr const char* COPYNO         = "COPYNO" ;
    static constexpr const char* SENSOR_ID      = "SENSOR_ID" ;
    static constexpr const char* SENSOR_INDEX   = "SENSOR_INDEX" ;
    static constexpr const char* REPEAT_INDEX   = "REPEAT_INDEX" ;
    static constexpr const char* REPEAT_ORDINAL = "REPEAT_ORDINAL" ;
    static constexpr const char* BOUNDARY       = "BOUNDARY" ;
    static constexpr const char* SENSOR_NAME    = "SENSOR_NAME" ;

    static int Idx(const char* attrib);
};

inline int snode_field::Idx(const char* attrib)
{
    int idx = -1 ;
    if(     0==strcmp(attrib,INDEX))          idx = 0 ;
    else if(0==strcmp(attrib,DEPTH))          idx = 1 ;
    else if(0==strcmp(attrib,SIBDEX))         idx = 2 ;
    else if(0==strcmp(attrib,PARENT))         idx = 3 ;
    else if(0==strcmp(attrib,NUM_CHILD))      idx = 4 ;
    else if(0==strcmp(attrib,FIRST_CHILD))    idx = 5 ;
    else if(0==strcmp(attrib,NEXT_SIBLING))   idx = 6 ;
    else if(0==strcmp(attrib,LVID))           idx = 7 ;
    else if(0==strcmp(attrib,COPYNO))         idx = 8 ;
    else if(0==strcmp(attrib,SENSOR_ID))      idx = 9 ;
    else if(0==strcmp(attrib,SENSOR_INDEX))   idx = 10 ;
    else if(0==strcmp(attrib,REPEAT_INDEX))   idx = 11 ;
    else if(0==strcmp(attrib,REPEAT_ORDINAL)) idx = 12 ;
    else if(0==strcmp(attrib,BOUNDARY))       idx = 13 ;
    else if(0==strcmp(attrib,SENSOR_NAME))    idx = 14 ;
    return idx ;
}


struct snode
{
    static constexpr const int NV = 15 ;

    int index ;        // 0
    int depth ;        // 1
    int sibdex ;       // 2    0-based sibling index
    int parent ;       // 3

    int num_child ;    // 4
    int first_child ;  // 5
    int next_sibling ; // 6
    int lvid ;         // 7

    int copyno ;       // 8
    int sensor_id ;    // 9  : -1 signifies "not-a-sensor"
    int sensor_index ; // 10
    int repeat_index ; // 11

    int repeat_ordinal ; // 12
    int boundary ;       // 13
    int sensor_name ;    // 14

    std::string desc() const ;
    int    get_attrib(int idx) const ;
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
       << " bd:" << std::setw(3) << boundary
       << " sn:" << std::setw(2) << sensor_name
       ;
    std::string str = ss.str();
    return str ;
}
inline int snode::get_attrib(int idx) const
{
    const int* aa = reinterpret_cast<const int*>(this) ;
    return idx > -1 && idx < NV ? aa[idx] : -1 ;
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

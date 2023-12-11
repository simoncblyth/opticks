#pragma once

#include "sseq.h"
#include "NPX.h"


struct sseq_index
{
    std::vector<sseq> q ;  // typically large input array 
    std::map<sseq, std::pair<int,int>> m ; 
    std::vector<sseq_unique> u ; 

    sseq_index( const NP* seq); 

    void load_seq( const NP* seq ); 
    void count_unique(); 
    void order_seq();
    std::string desc(int min_count=0) const; 
}; 


inline sseq_index::sseq_index( const NP* seq)
{
    load_seq(seq); 
    count_unique(); 
    order_seq(); 
}


inline void sseq_index::load_seq(const NP* seq)
{
    NPX::VecFromArray<sseq>(q, seq ); 
}

/**
sseq_index::count_unique
--------------------------------

Iterate over the source vector populating the 
map with the index of first occurrence and
count of the frequencey of occurrence.  

**/

inline void sseq_index::count_unique()
{ 
    for (int i = 0; i < int(q.size()); i++) 
    {
        const sseq& seq = q[i];
        std::map<sseq, std::pair<int,int>>::iterator it  = m.find(seq);
        if(it == m.end()) 
        {
            m[seq] = std::pair<int,int>(i, 1) ;
        }
        else 
        {
            m[seq] = std::pair<int,int>( it->second.first, 1+it->second.second );
        }
    }
}

/**
sseq_index::order_seq
-----------------------

1. copy from map m into vector u 
2. sort the u vector into descending count order

**/


inline void sseq_index::order_seq()
{
    for(auto it=m.begin() ; it != m.end() ; it++) u.push_back( { it->first, it->second.first, it->second.second } );  

    auto order = [](const sseq_unique& a, const sseq_unique& b) { return a.count > b.count ; } ; 
    std::sort( u.begin(), u.end(), order  ); 
}

inline std::string sseq_index::desc(int min_count) const
{
    std::stringstream ss ; 
    int num = u.size(); 
    ss << "[sseq_index::desc num " << num << std::endl ; 
    for(int i=0 ; i < num ; i++) 
    {
        if( u[i].count < min_count ) break ; 
        ss << u[i].desc() << std::endl ; 
    }
    ss << "]sseq_index::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}



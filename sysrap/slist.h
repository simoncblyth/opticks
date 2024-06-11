#pragma once

#include <string>
#include <vector>
#include <algorithm>

struct slist
{
    static bool Contains(const std::vector<int>& idx, int lvid );  
    static void FindIndices(std::vector<int>& idx, const std::vector<std::string>& name, const char* q ); 
    static int FindIndex(const std::vector<std::string>& name, const char* q );
}; 


inline bool slist::Contains(const std::vector<int>& idx, int lvid )
{
    return std::find( idx.begin(), idx.end(), lvid ) != idx.end() ; 
}

inline void slist::FindIndices(std::vector<int>& idx, const std::vector<std::string>& name, const char* q ) // static
{
    unsigned num_name = name.size();  
    for(unsigned i=0 ; i < num_name ; i++)
    {
        const char* n = name[i].c_str() ; 
        if(strcmp(q,n) == 0) idx.push_back(i) ; 
    } 
} 
inline int slist::FindIndex(const std::vector<std::string>& name, const char* q ) // static
{
    std::vector<int> idx ; 
    FindIndices(idx, name, q ); 
    return idx.size() == 1 ? idx[0] : -1 ; 
}



#include "SCount.hh"

#include <iomanip>
#include <sstream>

void SCount::add(int idx)
{
    if( mii.count(idx) == 0 ) mii[idx] = 0 ; 
    mii[idx] += 1 ;   
}

/**
SCount::is_all
---------------

Returns true when all the counts match the argument provides
**/

bool SCount::is_all(int count) const
{
    typedef std::map<int,int>::const_iterator IT ; 
    for(IT it=mii.begin() ; it != mii.end() ; it++ ) if(it->second != count) return false ;
    return true ;     
}


std::string SCount::desc() const 
{
    std::stringstream ss ; 
    typedef std::map<int,int>::const_iterator IT ; 
    bool all_same_count = false ; 
    int count = -1 ; 

    for(IT it=mii.begin() ; it != mii.end() ; it++ )
    {   
         int idx = it->first ; 

         count = it->second ; 
         all_same_count = is_all(count); 

         if(!all_same_count)
         {
             ss  
                << " " 
                << idx
                << ":" 
                << count
                ;   
         }
         else
         {
            ss << " " << idx ;   
         }
    }   

    if( all_same_count ) ss << " all_same_count " << count ; 
    std::string s = ss.str(); 
    return s ; 
}



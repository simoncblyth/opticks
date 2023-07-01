#pragma once
/** 
sidxname.h
===========

Typical use with std::set::

    sidxname d(-1, "0123456789abcdef0123456789abcdef") ; 
    sidxname e(-2, "0123456789abcdef0123456789abcdef_") ; // beyond 32 gets truncated

    std::set<sidxname,sidxname_ordering> mm ; 
    mm.insert(d);  
    mm.insert(e);  
 

**/

#include <cassert>
#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <iomanip>

struct sidxname
{ 
   int idx ;  
   char name[32] ; 

   sidxname(int idx_, const char* n) ;  

   std::string get_name() const ; 
   std::string desc() const ; 
};  


struct sidxname_ordering 
{
    bool operator()(const sidxname& lhs, const sidxname& rhs) const
    {
        return lhs.idx < rhs.idx ; 
    }
};


inline sidxname::sidxname(int idx_, const char* n )
   :
   idx(idx_)
{
   int max = sizeof(name) - 1 ; 
   bool ok = int(strlen(n)) <= max ; 
   if(!ok) std::cerr 
        << "sidxname::sidxname"
        << " strlen(n) " << strlen(n)
        << " max " << max 
        << " FATAL name is too long [" << n << "]" 
        << std::endl 
        ; 

   assert(ok) ; 
   strcpy( name, n );  
}


inline std::string sidxname::desc() const 
{ 
    std::stringstream ss ; 
    ss << std::setw(3) << idx << " : [" << name << "]" ; 
    std::string str = ss.str(); 
    return str ;
}


   

#pragma once


#ifdef DEBUG_RECORD

#include <vector>
#include <string>
#include "OpticksCSG.h"
#include "csg_classify.h"

struct quad4 ; 

/**
CSGRecord
===========

* CSGRecord_ENABLED envvar is the initial setting, this can be changed with SetEnabled. 
* operation also requires the special (non-default) compilation flag DEBUG_RECORD

**/

struct CSGRecord
{
    static bool ENABLED ;  
    static void SetEnabled(bool enabled); 

    static std::vector<quad4> record ;     


    CSGRecord( const quad4& r_ ); 
    const quad4& r ; 

    unsigned typecode ; 
    IntersectionState_t l_state ; 
    IntersectionState_t r_state ; 

    bool leftIsCloser ; 
    bool l_promote_miss ; 
    bool l_complement ; 
    bool l_unbounded ;
    bool l_false ; 
    bool r_promote_miss ; 
    bool r_complement ; 
    bool r_unbounded ;
    bool r_false ; 
    unsigned zero0 ; 
    unsigned zero1 ; 
    unsigned ctrl ; 


    static void Dump(const char* msg="CSGRecord::Dump"); 
    static std::string Desc( const quad4& r, unsigned irec, const char* label  ); 

    std::string desc(unsigned irec, const char* label  ) const ; 
    std::string desc_q2() const ; 
    std::string desc_q3() const ;
 
    static void Save(const char* dir); 
    static void Clear(); 
};

#endif


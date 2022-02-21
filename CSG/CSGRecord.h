#pragma once


#ifdef DEBUG_RECORD

#include <vector>
#include <string>
#include "OpticksCSG.h"
#include "csg_classify.h"

struct quad6 ; 

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

    static std::vector<quad6> record ;     


    CSGRecord( const quad6& r_ ); 
    const quad6& r ; 

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
    unsigned tloop ; 
    unsigned nodeIdx ; 
    unsigned ctrl ; 

    float tmin ;   // may be advanced : but dont see that with simple looping of leaf
    float t_min ;  // overall fixed value 
    float tminAdvanced ; //  direct collection of the advanced for upcoming looping 

    void unpack(unsigned packed ); 

    static void Dump(const char* msg="CSGRecord::Dump"); 
    static std::string Desc( const quad6& r, unsigned irec, const char* label  ); 
    std::string desc(unsigned irec, const char* label  ) const ; 
    static void Save(const char* dir); 
    static void Clear(); 
};

#endif


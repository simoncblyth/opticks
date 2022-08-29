#pragma once


#ifdef DEBUG_RECORD

#include <vector>
#include <string>
#include "OpticksCSG.h"
#include "csg_classify.h"
#include "CSG_API_EXPORT.hh"

struct quad6 ; 

/**
CSGRecord
===========

* CSGRecord_ENABLED envvar is the initial setting, this can be changed with SetEnabled. 
* operation also requires the special (non-default) compilation flag DEBUG_RECORD


For understanding CSGRecords note that records are added at CSG decisions 
For example with the below tree decisions are made at nodeIdx 2,3,1
as postorder is followed, ie 4,5,[2],6,7,[3],[1] 
But loopbacks are done so there will often be repeated decisions
with tmin advanced also at nodeIdx 2,3,1.


                               in                                                         
                              1                                                           
                                 0.00                                                     
                                -0.00                                                     
                                                                                          
           un                                      in                                     
          2                                       3                                       
             0.00                                    0.00                                 
            -0.00                                   -0.00                                 
                                                                                          
 zs                  cy                 !zs                 !cy                           
4                   5                   6                   7                             
 194.00                0.10              186.00                0.10                       
 -39.00              -38.90              -40.00              -39.90                       
                                                                                          


TODO: get the winning nodeIdx into the final result CSGRecord ?

**/

struct CSG_API CSGRecord
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


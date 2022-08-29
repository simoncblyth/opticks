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


/**
Attempting to read the CSGRecord
==================================

* TODO: confirm the below reading of the tealeaves by for example carrying forward winner nodeIdx thru the levels 
* TODO: make it easier to read by removing noise or improving layout 
* TODO: reviviing the recursive CPU only intersect implementation would be useful as a cross-check
  and for elucidation as the recursive algorithm is much simpler and should give exactly the same results 
  using exactly the same node visits 

  * optixrap/cu/csg_intersect_boolean.h:unsupported_recursive_csg_r




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
 


2022-08-29 09:57:04.550 INFO  [40471266] [CSGSimtraceRerun::report@169] with : DEBUG_RECORD 
2022-08-29 09:57:04.550 INFO  [40471266] [CSGRecord::Dump@134] CSGSimtraceRerun::report CSGRecord::record.size 10IsEnabled 0
 tloop    0 nodeIdx    2 irec          0 label                                                                                        rec union
                 r.q0.f left  (    0.0000    0.0000   -1.0000  -78.8386) Enter  - - - leftIsCloser
                r.q1.f right  (    0.0000    0.0000   -1.0000   78.9570) Enter  - - -   ctrl RETURN_A
           r.q3.f tmin/t_min  (    0.0000    0.0000    0.0000    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (    0.0000    0.0000   -1.0000  -78.8386) 
 tloop    0 nodeIdx    3 irec          1 label                                                                                        rec intersection
                 r.q0.f left  (   -0.0000   -0.0000    1.0000  -77.6548) Exit  - l_complement l_unbounded leftIsCloser
                r.q1.f right  (   -0.0000   -0.0000    1.0000   77.7732) Exit  - r_complement r_unbounded   ctrl RETURN_A
           r.q3.f tmin/t_min  (    0.0000    0.0000    0.0000    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (   -0.0000   -0.0000    1.0000   77.6548) 
 tloop    0 nodeIdx    1 irec          2 label                                                                                        rec intersection
                 r.q0.f left  (    0.0000    0.0000   -1.0000  -78.8386) Enter  - - -  
                r.q1.f right  (   -0.0000   -0.0000    1.0000   77.6548) Exit  - r_complement r_unbounded rightIsCloser ctrl LOOP_B
           r.q3.f tmin/t_min  (    0.0000    0.0000   77.6549    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced    77.6549
               r.q4.f result  (    0.0000    0.0000    0.0000    0.0000) 

Third CSGRecord is a root (nodeIdx:1) decision to LOOP_B, which is nodeIdx:3 


 tloop    1 nodeIdx    3 irec          3 label                                                                                        rec intersection
                 r.q0.f left  (   -0.0000    0.0000    0.0000   -0.0000) Exit  l_promote_miss l_complement -  
                r.q1.f right  (   -0.0000   -0.0000    1.0000   77.7732) Exit  - r_complement r_unbounded rightIsCloser ctrl RETURN_B
           r.q3.f tmin/t_min  (   77.6549    0.0000    0.0000    0.0000)  tmin    77.6549 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (   -0.0000   -0.0000    1.0000   77.7732) 
 tloop    2 nodeIdx    1 irec          4 label                                                                                        rec intersection
                 r.q0.f left  (    0.0000    0.0000   -1.0000  -78.8386) Enter  - - -  
                r.q1.f right  (   -0.0000   -0.0000    1.0000   77.7732) Exit  - r_complement r_unbounded rightIsCloser ctrl LOOP_B
           r.q3.f tmin/t_min  (    0.0000    0.0000   77.7733    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced    77.7733
               r.q4.f result  (    0.0000    0.0000    0.0000    0.0000) 

Get back to root but get another LOOP_B decision, so back to nodeIdx:3


 tloop    3 nodeIdx    3 irec          5 label                                                                                        rec intersection
                 r.q0.f left  (   -0.0000    0.0000    0.0000   -0.0000) Exit  l_promote_miss l_complement -  
                r.q1.f right  (   -0.0000   -0.0000   -1.0000  125.1237) Enter  - r_complement r_unbounded rightIsCloser ctrl RETURN_B
           r.q3.f tmin/t_min  (   77.7733    0.0000    0.0000    0.0000)  tmin    77.7733 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (   -0.0000   -0.0000   -1.0000  125.1237) 
 tloop    4 nodeIdx    1 irec          6 label                                                                                        rec intersection
                 r.q0.f left  (    0.0000    0.0000   -1.0000  -78.8386) Enter  - - - leftIsCloser
                r.q1.f right  (   -0.0000   -0.0000   -1.0000  125.1237) Enter  - r_complement r_unbounded   ctrl LOOP_A
           r.q3.f tmin/t_min  (    0.0000    0.0000   78.8387    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced    78.8387
               r.q4.f result  (    0.0000    0.0000    0.0000    0.0000) 

This root decision is to LOOP_A, so back to nodeIdx:2 


 tloop    5 nodeIdx    2 irec          7 label                                                                                        rec union
                 r.q0.f left  (    0.0138    0.0000    0.9998 -354.6146) Exit  - - -  
                r.q1.f right  (    0.0000    0.0000   -1.0000   78.9570) Enter  - - - rightIsCloser ctrl LOOP_B
           r.q3.f tmin/t_min  (   78.8387    0.0000   78.9571    0.0000)  tmin    78.8387 t_min     0.0000 tminAdvanced    78.9571
               r.q4.f result  (    0.0000    0.0000    0.0000    0.0000) 

First LOOP_B so are checking again the outer cy, but the next decision is RETURN_A so the outer zs wins at this stage

 tloop    7 nodeIdx    2 irec          8 label                                                                                        rec union
                 r.q0.f left  (    0.0138    0.0000    0.9998 -354.6146) Exit  - - -  
                r.q1.f right  (    0.0000    0.0000    1.0000  125.1237) Exit  - - - rightIsCloser ctrl RETURN_A
           r.q3.f tmin/t_min  (   78.8387    0.0000    0.0000    0.0000)  tmin    78.8387 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (    0.0138    0.0000    0.9998 -354.6146) 


Notice how the same distance : 125.1237 is appearing on both sides of the tree :that is the cause of coincidence issues
and probably the reason for all the loopbacks ?

But then back to root comparison get final RETURN_B decision, so 

 tloop    8 nodeIdx    1 irec          9 label                                                                                        rec intersection
                 r.q0.f left  (    0.0138    0.0000    0.9998 -354.6146) Exit  - - -  
                r.q1.f right  (   -0.0000   -0.0000   -1.0000  125.1237) Enter  - r_complement r_unbounded rightIsCloser ctrl RETURN_B
           r.q3.f tmin/t_min  (    0.0000    0.0000    0.0000    0.0000)  tmin     0.0000 t_min     0.0000 tminAdvanced     0.0000
               r.q4.f result  (   -0.0000   -0.0000   -1.0000  125.1237) 
2022-08-29 09:57:04.551 INFO  [40471266] [CSGSimtraceRerun::report@172]  save CSGRecord.npy to fold /tmp/blyth/opticks/GeoChain/nmskSolidMask/G4CXSimtraceTest/ALL
2022-08-29 09:57:04.551 INFO  [40471266] [CSGRecord::Save@247]  dir /tmp/blyth/opticks/GeoChain/nmskSolidMask/G4CXSimtraceTest/ALL num_record 10
NP::init size 240 ebyte 4 num_char 960
with : DEBUG 
epsilon:CSG blyth$ 


**/


#pragma once
/**
SCenterExtentGenstep.hh : TODO REPLACE THIS WITH SFrameGenstep
=================================================================

NB this was demoted from extg4/X4Intersect as its all generally applicable 
and hence belongs at a lower level 

Used from::

   CSG/CSGGeometry.cc
   extg4/X4Intersect.cc

HMM: compare this with CSG/CSGGenstep and consolidate to avoid duplication 

* CSG/CSGGenstep has been removed from the build and will be deleted
* TODO: compare SCenterExtentGenstep to SFrameGenstep and consolidate

  * probably duplication arose from separate dev for Geant4 and Opticks sides


**/

struct NP ; 
struct quad6 ; 
struct float4 ; 
template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"



struct SYSRAP_API SCenterExtentGenstep
{
    static const char* BASE ; 
    static const plog::Severity LEVEL ; 

    SCenterExtentGenstep(const float4* ce_=nullptr); 
    void init(); 
    const char* desc() const ; 
    void save(const char* dir) const ; 
    void save() const ; 

    void save_vec(const char* dir, const char* name, const std::vector<quad4>& vv ) const ; 

    template<typename T> void set_meta(const char* key, T value ) ; 


    NP*    gs ;         // not const as need to externally set the meta 
    float  gridscale ;   

    quad4* peta ; 
    bool   dump ; 
    float4 ce ;

    std::vector<int> cegs ; 
    int nx ; 
    int ny ; 
    int nz ; 

    std::vector<int> override_ce ;   

    std::vector<quad4> pp ;
    std::vector<quad4> ii ;

};




#pragma once
/**
X4Intersect.hh : NB ARE IN PROCESS OF REMOVING THIS, REPLACE WITH sysrap/SIntersect.h
========================================================================================

::

    epsilon:extg4 blyth$ opticks-fl X4Intersect 
    ./ana/pub.py
    ./ana/axes.py
    ./CSG/tests/CSGIntersectSolidTest.py
    ./extg4/xxs.sh
    ./extg4/x4t.sh
    ./extg4/CMakeLists.txt
    ./extg4/X4Intersect.hh
    ./extg4/xxv.sh
    ./extg4/X4Intersect.cc
    ./extg4/tests/X4IntersectSolidTest.py
    ./extg4/tests/CMakeLists.txt
    ./extg4/tests/X4IntersectVolumeTest.py
    ./extg4/tests/X4IntersectVolumeTest.cc
    ./extg4/tests/X4_Get.hh
    ./extg4/tests/X4IntersectSolidTest.cc
    ./extg4/pubv.sh
    ./extg4/X4Simtrace.hh
    ./extg4/pub.sh
    ./GeoChain/tests/GeoChainSolidTest.cc
    ./sysrap/SFrameGenstep.hh
    ./sysrap/SVolume.h
    ./sysrap/SCenterExtentGenstep.hh
    epsilon:opticks blyth$ 


**/

#include <vector>
#include <string>
#include "plog/Severity.h"

#include "X4_API_EXPORT.hh"
#include "G4ThreeVector.hh"
#include "geomdefs.hh"

struct float4 ; 
struct SCenterExtentGenstep ; 
class G4VSolid ; 
class G4MultiUnion ;

struct X4_API X4Intersect
{
    static const plog::Severity LEVEL ;  
    static const bool VERBOSE ;  
    static void Scan(const G4VSolid* solid, const char* name, const char* basedir ); 

    X4Intersect( const G4VSolid* solid_ ); 
    const char* desc() const ; 

    static double Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump); 
    static double Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in  ); 
    static double DistanceMultiUnionNoVoxels_(const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in );

    void init(); 
    void scan_(); 
    void scan(); 

    const G4VSolid* solid ; 
    float4*         ce ; 
    SCenterExtentGenstep* cegs ; 
}; 


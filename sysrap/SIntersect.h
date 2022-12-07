#pragma once

#include <vector>
#include <string>

#include "plog/Severity.h"
#include "G4ThreeVector.hh"
#include "geomdefs.hh"

struct float4 ; 
struct SCenterExtentGenstep ; 
class G4VSolid ; 
class G4MultiUnion ;

struct SIntersect
{
    static constexpr const  plog::Severity LEVEL = info ;  
    static constexpr const bool VERBOSE = true ; 
 
    static void Scan(const G4VSolid* solid, const char* name, const char* basedir ); 

    SIntersect( const G4VSolid* solid_ ); 
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

#include <iostream>
#include <iomanip>

#include "G4ThreeVector.hh"
#include "G4VSolid.hh"
#include "G4MultiUnion.hh"

#include "scuda.h"
#include "squad.h"

#include "SSys.hh"
#include "SDirect.hh"   // cout_redirect cerr_redirect 
#include "SPath.hh"
#include "SCenterExtentGenstep.hh"


#include "sgeomdefs.h"
#include "ssolid.h"

#include "SLOG.hh"


/**
SIntersect::Scan
-------------------

Used from tests/SIntersectSolidTest.cc which is used by xxs.sh 

**/

inline void SIntersect::Scan(const G4VSolid* solid, const char* name, const char* basedir )  // static
{
    assert( solid && "SIntersect::Scan requires solid"); 

    SIntersect* si = new SIntersect(solid); 
    si->scan(); 

    const std::string& solidname = solid->GetName() ; 

    const char* outdir = SPath::Resolve(basedir, name, "SIntersect", DIRPATH );

    LOG(LEVEL) 
        << "si.desc " << si->desc() 
        << " solidname " << solidname.c_str() 
        << " name " << name 
        << " outdir " << outdir 
        ; 

    SCenterExtentGenstep* cegs = si->cegs ; 
    cegs->set_meta<std::string>("name", name); 
    cegs->set_meta<int>("iidx", 0 ); 
    cegs->save(outdir); 
}


inline SIntersect::SIntersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    ce(nullptr),   // float4*  
    cegs(nullptr)  // SCenterExtentGenstep*
{
    init(); 
}

/**
SIntersect::init
------------------

Uses G4VSolid::BoundingLimits to determine *ce* center and extent, 
and uses *ce* to create *cegs* SCenterExtentGenstep instance. 

**/

inline void SIntersect::init()
{
    ce = new float4 ; 
    ssolid::GetCenterExtent( *ce, solid ); 
    cegs = new SCenterExtentGenstep(ce) ; 
}

inline const char* SIntersect::desc() const 
{
    return cegs->desc() ; 
}


/**
SIntersect::scan_
-------------------

Using the *cegs.pp* vector of "photon" positions and directions
calulate distances to the solid.  Collect intersections
into *cegs.ii* vector. 

TODO: 

* adopt simtrace layout, even although some aspects like surface normal 
  will  be missing from it. This means cegs->pp and cegs->ii will kinda merge 

**/

inline void SIntersect::scan_()
{
    const std::vector<quad4>& pp = cegs->pp ; 
    std::vector<quad4>& ii = cegs->ii ; 

    bool dump = false ; 
    for(unsigned i=0 ; i < pp.size() ; i++)
    {
        const quad4& p = pp[i]; 

        G4ThreeVector pos(p.q0.f.x, p.q0.f.y, p.q0.f.z); 
        G4ThreeVector dir(p.q1.f.x, p.q1.f.y, p.q1.f.z); 

        G4double t = ssolid::Distance( solid, pos, dir, dump );  

        if( t == kInfinity ) continue ; 
        G4ThreeVector ipos = pos + dir*t ;  

        quad4 isect ; 
        isect.zero(); 

        isect.q0.f.x = float(ipos.x()) ;  
        isect.q0.f.y = float(ipos.y()) ;  
        isect.q0.f.z = float(ipos.z()) ;  
        isect.q0.f.w = float(t) ; 
        // TODO: normals, flags, ...

        ii.push_back(isect); 
    } 
}


inline void SIntersect::scan()
{
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());

       scan_(); 
    }   
    std::string cout_ = coutbuf.str() ; 
    std::string cerr_ = cerrbuf.str() ; 


    LOG(LEVEL) 
        << "scan" 
        << " cout " << strlen(cout_.c_str()) 
        << " cerr " << strlen(cerr_.c_str()) 
        ;

    if(VERBOSE)
    {
        bool with_cout = cout_.size() > 0 ; 
        bool with_cerr = cerr_.size() > 0 ; 
        LOG_IF(LEVEL, with_cout) << "cout from scan " << std::endl << cout_ ; 
        LOG_IF(LEVEL, with_cerr) << "cerr from scan " << std::endl << cerr_ ; 
    }
}


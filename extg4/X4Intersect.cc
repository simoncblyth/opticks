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

#include "X4Intersect.hh"

#include "x4geomdefs.h"
#include "x4solid.h"


#include "PLOG.hh"



const plog::Severity X4Intersect::LEVEL = PLOG::EnvLevel("X4Intersect", "DEBUG") ; 
const bool X4Intersect::VERBOSE = SSys::getenvbool("VERBOSE") ; 


/**
X4Intersect::Scan
-------------------

Used from tests/X4IntersectSolidTest.cc which is used by xxs.sh 

**/

void X4Intersect::Scan(const G4VSolid* solid, const char* name, const char* basedir )  // static
{
    assert( solid && "X4Intersect::Scan requires solid"); 

    X4Intersect* x4i = new X4Intersect(solid); 
    x4i->scan(); 

    const std::string& solidname = solid->GetName() ; 

    const char* outdir = SPath::Resolve(basedir, name, "X4Intersect", DIRPATH );

    LOG(LEVEL) 
        << "x4i.desc " << x4i->desc() 
        << " solidname " << solidname.c_str() 
        << " name " << name 
        << " outdir " << outdir 
        ; 

    SCenterExtentGenstep* cegs = x4i->cegs ; 
    cegs->set_meta<std::string>("name", name); 
    cegs->set_meta<int>("iidx", 0 ); 
    cegs->save(outdir); 
}


X4Intersect::X4Intersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    ce(nullptr),   // float4*  
    cegs(nullptr)  // SCenterExtentGenstep*
{
    init(); 
}

/**
X4Intersect::init
------------------

Uses G4VSolid::BoundingLimits to determine *ce* center and extent, 
and uses *ce* to create *cegs* SCenterExtentGenstep instance. 

**/

void X4Intersect::init()
{
    ce = new float4 ; 
    x4solid::GetCenterExtent( *ce, solid ); 
    cegs = new SCenterExtentGenstep(ce) ; 
}

const char* X4Intersect::desc() const 
{
    return cegs->desc() ; 
}


/**
X4Intersect::scan_
-------------------

Using the *cegs.pp* vector of "photon" positions and directions
calulate distances to the solid.  Collect intersections
into *cegs.ii* vector. 

TODO: 

* adopt simtrace layout, even although some aspects like surface normal 
  will  be missing from it. This means cegs->pp and cegs->ii will kinda merge 

**/

void X4Intersect::scan_()
{
    const std::vector<quad4>& pp = cegs->pp ; 
    std::vector<quad4>& ii = cegs->ii ; 

    bool dump = false ; 
    for(unsigned i=0 ; i < pp.size() ; i++)
    {
        const quad4& p = pp[i]; 

        G4ThreeVector pos(p.q0.f.x, p.q0.f.y, p.q0.f.z); 
        G4ThreeVector dir(p.q1.f.x, p.q1.f.y, p.q1.f.z); 

        G4double t = x4solid::Distance( solid, pos, dir, dump );  

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


void X4Intersect::scan()
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
        if(cout_.size() > 0) LOG(LEVEL) << "cout from scan " << std::endl << cout_ ; 
        if(cerr_.size() > 0) LOG(LEVEL) << "cerr from scan "  << std::endl << cerr_ ; 
    }
}


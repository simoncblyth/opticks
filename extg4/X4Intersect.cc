#include <iostream>
#include <iomanip>

#include "G4ThreeVector.hh"
#include "G4VSolid.hh"

#include "scuda.h"
#include "stran.h"
#include "squad.h"

#include "SPath.hh"
#include "SEvent.hh"
#include "SSys.hh"
#include "NP.hh"

#include "X4geomdefs.hh"
#include "X4Intersect.hh"
#include "PLOG.hh"


void X4Intersect::Scan(const G4VSolid* solid, const char* name, const char* basedir, const std::string& meta )  // static
{
    assert( solid && "X4Intersect::Scan requires solid"); 

    X4Intersect* x4i = new X4Intersect(solid); 
    x4i->scan(); 

    const std::string& solidname = solid->GetName() ; 

    int createdirs = 2 ; // 2:dirpath 
    const char* outdir = SPath::Resolve(basedir, name, "X4Intersect", createdirs);

    LOG(info) 
        << "x4i.desc " << x4i->desc() 
        << " solidname " << solidname.c_str() 
        << " name " << name 
        << " outdir " << outdir 
        ; 

    x4i->gs->meta = meta ; 
    x4i->save(outdir); 
}






X4Intersect::X4Intersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    gs(nullptr),
    gridscale(SSys::getenvfloat("GRIDSCALE", 1.0 )),
    dump(false)
{
    init(); 
}

const char* X4Intersect::desc() const 
{
    std::stringstream ss ; 
    ss << " CXS_CEGS (" ; 
    for(unsigned i=0 ; i < cegs.size() ; i++ ) ss << cegs[i] << " " ; 
    ss << ")" ; 

    ss << " GRIDSCALE " << gridscale ; 
    ss << " CE (" 
       << ce.x << " " 
       << ce.y << " " 
       << ce.z << " " 
       << ce.w 
       << ") " 
       ;

    ss << " gs " << gs->sstr() ; 
    ss << " pp " << pp.size() ; 
    ss << " ii " << ii.size() ; 

    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


void X4Intersect::init()
{
    LOG(info) << "[ gridscale " << gridscale  ; 

    ce = make_float4(0.f, 0.f, 0.f, 100.f ); 

    SSys::getenvintvec("CXS_CEGS", cegs, ':', "16:0:9:10" );  
    // expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 

    SEvent::StandardizeCEGS(ce, cegs, gridscale );  
    assert( cegs.size() == 7 );  


    SSys::getenvintvec("CXS_OVERRIDE_CE",  override_ce, ':', "0:0:0:0" );  

    const Tran<double>* geotran = Tran<double>::make_identity(); 

    if( override_ce.size() == 4 && override_ce[3] > 0 ) 
    {   
        ce.x = float(override_ce[0]); 
        ce.y = float(override_ce[1]); 
        ce.z = float(override_ce[2]); 
        ce.w = float(override_ce[3]); 
        LOG(info) << "override ce with CXS_OVERRIDE_CE (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;   
    }   

    gs = SEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran );  
   
    SEvent::GenerateCenterExtentGenstepsPhotons( pp, gs );  

    LOG(info) << "]" ; 
}



G4double X4Intersect::Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ) // static
{
    EInside in =  solid->Inside(pos) ; 
    G4double t = kInfinity ; 
    switch( in )
    {
        case kInside:  t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kSurface: t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kOutside: t = solid->DistanceToIn(  pos, dir ) ; break ; 
        default:  assert(0) ; 
    }

    if(dump && t != kInfinity)
    {
        std::cout 
            << " pos " 
            << "(" 
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.x() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.y() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << pos.z() 
            << ")"
            << " dir " 
            << "(" 
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.x() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.y() << " "
            << std::fixed << std::setw(10) << std::setprecision(3) << dir.z() 
            << ")"
            << " in " << X4geomdefs::EInside_(in ) 
            ;

       if( t == kInfinity)
       {  
            std::cout 
                << " t " << std::setw(10) << "kInfinity" 
                << std::endl 
                ; 
       }
       else
       {
           G4ThreeVector ipos = pos + dir*t ;  
           std::cout 
                << " t " << std::fixed << std::setw(10) << std::setprecision(3) << t 
                << " ipos " 
                << "(" 
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.x() << " "
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.y() << " "
                << std::fixed << std::setw(10) << std::setprecision(3) << ipos.z() 
                << ")"
                << std::endl 
                ; 
       }
    }
    return t ; 
}


/**
X4Intersect::scan
------------------

Using the *pp* vector of "photon" positions and directions
calulate distances to the solid.  Collect intersections
into *ss* vector. 

TODO: collect surface normals 

**/

void X4Intersect::scan()
{
    for(unsigned i=0 ; i < pp.size() ; i++)
    {
        const quad4& p = pp[i]; 

        G4ThreeVector pos(p.q0.f.x, p.q0.f.y, p.q0.f.z); 
        G4ThreeVector dir(p.q1.f.x, p.q1.f.y, p.q1.f.z); 

        G4double t = Distance( solid, pos, dir, dump );  

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

void X4Intersect::save(const char* dir) const 
{
    LOG(info) << "[ ii.size " << ii.size() ; 
    NP* a = NP::Make<float>(ii.size(), 4, 4); 
    LOG(info) << a->sstr() ;    
    a->read<float>((float*)ii.data()); 
    a->save(dir, "isect.npy");  
    gs->save(dir, "gs.npy" ); 

    LOG(info) << "]" ; 
}




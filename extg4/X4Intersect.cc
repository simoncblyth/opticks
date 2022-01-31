#include <iostream>
#include <iomanip>

#include "G4ThreeVector.hh"
#include "G4VSolid.hh"
#include "G4MultiUnion.hh"

#include "scuda.h"
#include "squad.h"

#include "SDirect.hh"   // cout_redirect cerr_redirect 
#include "SPath.hh"
#include "SCenterExtentGenstep.hh"

#include "X4geomdefs.hh"
#include "X4Intersect.hh"
#include "PLOG.hh"


void X4Intersect::Scan(const G4VSolid* solid, const char* name, const char* basedir )  // static
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

    SCenterExtentGenstep* cegs = x4i->cegs ; 
    cegs->set_meta<std::string>("name", name); 
    cegs->set_meta<int>("iidx", 0 ); 
    cegs->save(outdir); 
}


X4Intersect::X4Intersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    cegs(new SCenterExtentGenstep)
{
}

const char* X4Intersect::desc() const 
{
    return cegs->desc() ; 
}


G4double X4Intersect::Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
{
    in =  solid->Inside(pos) ; 
    G4double t = kInfinity ; 
    switch( in )
    {
        case kInside:  t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kSurface: t = solid->DistanceToOut( pos, dir ) ; break ; 
        case kOutside: t = solid->DistanceToIn(  pos, dir ) ; break ; 
        default:  assert(0) ; 
    }
    return t ; 
}

G4double X4Intersect::DistanceMultiUnionNoVoxels_(const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in ) // static
{
    in =  solid->InsideNoVoxels(pos) ; 
    G4double t = kInfinity ; 
    switch( in )
    {
        case kInside:  t = solid->DistanceToOutNoVoxels( pos, dir, nullptr ) ; break ; 
        case kSurface: t = solid->DistanceToOutNoVoxels( pos, dir, nullptr ) ; break ; 
        case kOutside: t = solid->DistanceToInNoVoxels(  pos, dir ) ; break ; 
        default:  assert(0) ; 
    }
    return t ; 
}


G4double X4Intersect::Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump ) // static
{
    EInside in ; 

    const G4MultiUnion* m = dynamic_cast<const G4MultiUnion*>(solid) ; 
    G4double t = m ? DistanceMultiUnionNoVoxels_(m, pos, dir, in ) : Distance_( solid, pos, dir, in  );  

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


    LOG(info) 
        << "scan" 
        << " cout " << strlen(cout_.c_str()) 
        << " cerr " << strlen(cerr_.c_str()) 
        ;

    /*
    if(cout_.size() > 0) LOG(info) << "cout from scan " << std::endl << cout_ ; 
    if(cerr_.size() > 0) LOG(warning) << "cerr from scan "  << std::endl << cerr_ ; 
    */
}




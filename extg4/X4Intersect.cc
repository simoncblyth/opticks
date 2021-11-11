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

X4Intersect::X4Intersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    gs(nullptr),
    gridscale(SSys::getenvfloat("GRIDSCALE", 1.0 )),
    dump(true)
{
    init(); 
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


void X4Intersect::scan()
{
    for(unsigned i=0 ; i < pp.size() ; i++)
    {
        const quad4& p = pp[i]; 

        G4ThreeVector pos(p.q0.f.x, p.q0.f.y, p.q0.f.z); 
        G4ThreeVector dir(p.q1.f.x, p.q1.f.y, p.q1.f.z); 

        EInside in =  solid->Inside(pos) ; 
        G4double t = ( in == kInside || in == kSurface ) ? solid->DistanceToOut( pos, dir ) : solid->DistanceToIn( pos, dir ) ; 
        if( t == kInfinity ) continue ; 

        if(dump) std::cout 
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
            if(dump) std::cout 
                << " t " << std::setw(10) << "kInfinity" 
                << std::endl 
                ; 
       }
       else
       {
           G4ThreeVector ipos = pos + dir*t ;  

           quad4 s ; 
           s.zero(); 

           s.q0.f.x = float(ipos.x()) ;  
           s.q0.f.y = float(ipos.y()) ;  
           s.q0.f.z = float(ipos.z()) ;  
           s.q0.f.w = float(t) ; 

           ss.push_back(s); 

           if(dump) std::cout 
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
}

void X4Intersect::save(const char* dir) const 
{
    LOG(info) << "[ ss.size " << ss.size() ; 
    NP* a = NP::Make<float>(ss.size(), 4, 4); 
    LOG(info) << a->sstr() ;    
    float* data = (float*)ss.data(); 
    a->read<float>(data); 
    a->save(dir, "isect.npy");  
    gs->save(dir, "gs.npy" ); 

    LOG(info) << "]" ; 
}


void X4Intersect::Scan(const G4VSolid* solid, const char* basedir )  // static
{
    X4Intersect* x4i = new X4Intersect(solid); 
    x4i->scan(); 

    const std::string& name = solid->GetName() ; 
    int createdirs = 2 ; // 2:dirpath 
    const char* outdir = SPath::Resolve(basedir, name.c_str(), createdirs);
    LOG(info) << " outdir " << outdir ; 

    x4i->save(outdir); 
}




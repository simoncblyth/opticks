#include <iostream>
#include <iomanip>

#include "G4ThreeVector.hh"
#include "G4VSolid.hh"

#include "scuda.h"
#include "stran.h"
#include "squad.h"

#include "SEvent.hh"
#include "SSys.hh"

#include "X4geomdefs.hh"
#include "X4Intersect.hh"
#include "PLOG.hh"

X4Intersect::X4Intersect( const G4VSolid* solid_  )
    :
    solid(solid_), 
    gs(nullptr),
    gridscale(1.),
    dump(true)
{
    init(); 
}

void X4Intersect::init()
{
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
           s.q0.f.x = ipos.x() ;  
           s.q0.f.y = ipos.y() ;  
           s.q0.f.z = ipos.z() ;  
           s.q0.f.z = t ; 

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



#include <cassert>

#include "scuda.h"
#include "squad.h"
#include "sc4u.h"
#include "OpticksGenstep.h"
#include "NP.hh"

#include "SGenstep.hh"

const char* SGenstep::XYZ_ = "XYZ" ; 
const char* SGenstep::YZ_  = "YZ" ; 
const char* SGenstep::XZ_  = "XZ" ; 
const char* SGenstep::XY_  = "XY" ; 

const char* SGenstep::GridAxesName( int gridaxes)  // static 
{
    const char* s = nullptr ; 
    switch( gridaxes )
    {
        case XYZ: s = XYZ_ ; break ; 
        case YZ:  s = YZ_  ; break ; 
        case XZ:  s = XZ_  ; break ; 
        case XY:  s = XY_  ; break ; 
    }
    return s ;
}

/**
SGenstep::GridAxes
--------------------------

The nx:ny:nz dimensions of the grid are used to classify it into::

    YZ 
    XZ  
    XY 
    XYZ

For a planar grid one of the nx:ny:nz grid dimensions is zero.
XYZ is a catch all for non-planar grids.

**/

int SGenstep::GridAxes(int nx, int ny, int nz)  // static
{
    int gridaxes = XYZ ;  
    if( nx == 0 && ny > 0 && nz > 0 )
    {
        gridaxes = YZ ;  
    }
    else if( nx > 0 && ny == 0 && nz > 0 )
    {
        gridaxes = XZ ;  
    }
    else if( nx > 0 && ny > 0 && nz == 0 )
    {
        gridaxes = XY ;  
    }
    return gridaxes ; 
}

/**
SGenstep::GenstepID
-------------------

Pack four signed integers (assumed to be in char range -128 to 127) 
into a 32 bit unsigtned char using C4U uniform.  

**/

unsigned SGenstep::GenstepID( int ix, int iy, int iz, int iw )
{ 
    C4U gsid ;   // sc4u.h 

    gsid.c4.x = ix ; 
    gsid.c4.y = iy ; 
    gsid.c4.z = iz ; 
    gsid.c4.w = iw ; 

    return gsid.u ; 
}




/**
SGenstep::ConfigureGenstep
---------------------------

TODO: pack enums to make room for a photon_offset 

* gsid was MOVED from (1,3) to (0,2) when changing genstep to carry transform

**/

void SGenstep::ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep )
{
    assert( gencode == OpticksGenstep_TORCH ); 
    assert( gridaxes == XYZ ||  gridaxes == YZ || gridaxes == XZ || gridaxes == XY ); 

    gs.q0.i.x = gencode ;
    gs.q0.i.y = gridaxes ; 
    gs.q0.u.z = gsid ;     
    gs.q0.i.w = photons_per_genstep ;
}

NP* SGenstep::MakeArray(const std::vector<quad6>& gs ) // static 
{
    assert( gs.size() > 0); 
    NP* a = NP::Make<float>( gs.size(), 6, 4 );  
    a->read2<float>( (float*)gs.data() );  
    return a ; 
}





#include <cassert>

#include "G4OpticalSurface.hh"   
#include "G4LogicalBorderSurface.hh"   

#include "X4LogicalBorderSurface.hh"

#include "GBorderSurface.hh"   
#include "PLOG.hh"


GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* lbs)
{
    X4LogicalBorderSurface xlbs(lbs);
    GBorderSurface* bs = xlbs.getBorderSurface();
    return bs ; 
}


X4LogicalBorderSurface::X4LogicalBorderSurface(const G4LogicalBorderSurface* lbs)
    :
    m_lbs(lbs),
    m_bs(NULL)
{
    init();
}  

GBorderSurface* X4LogicalBorderSurface::getBorderSurface() const 
{
    return m_bs ;  
}

void X4LogicalBorderSurface::init()
{
    //  hmm this is common to skin and border, coming from base G4LogicalSurface

    const G4LogicalSurface* lsurf = m_lbs ; 
    const G4SurfaceProperty*  psurf = lsurf->GetSurfaceProperty() ;   
    const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
    assert( opsurf );   

    LOG(info) << " opsurf " << opsurf->GetName() ; 

 

}




#include "CFG4_BODY.hh"
#include <algorithm>
#include <sstream>
#include <vector>

#include "CFG4_PUSH.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"
#include "PLOG.hh"
#include "CCheck.hh"


CCheck::CCheck(Opticks* ok, G4VPhysicalVolume* top) 
   :
   m_ok(ok),
   m_top(top)
{
}


void CCheck::checkSurf()
{
    G4int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
    LOG(info) << "CCheck::checkSurf" 
              << " NumberOfBorderSurfaces " << nsurf
              ; 
    
    assert(m_top); 
    G4LogicalVolume* lv = m_top->GetLogicalVolume() ;
    checkSurfTraverse( lv, 0 ); 
}


// after G4GDMLWriteStructure::GetBorderSurface
const G4LogicalBorderSurface* CCheck::GetBorderSurface(const G4VPhysicalVolume* const pvol)
{
  G4LogicalBorderSurface* surf = 0;
  G4int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
  if (nsurf)
  {
    const G4LogicalBorderSurfaceTable* btable =
          G4LogicalBorderSurface::GetSurfaceTable();
    std::vector<G4LogicalBorderSurface*>::const_iterator pos;
    for (pos = btable->begin(); pos != btable->end(); pos++)
    {   
      if (pvol == (*pos)->GetVolume1())  // just the first in the couple 
      {                                  // is enough
        surf = *pos; break;
      }   
    }   
  }
  return surf;
}


void CCheck::checkSurfTraverse(const G4LogicalVolume* const lv, const int depth)
{
    const G4int daughterCount = lv->GetNoDaughters();    

    std::cout 
          << " checkSurfTraverse "
          << " depth " << depth
          << " daughterCount " << daughterCount
          << " lv " << lv->GetName() 
          << std::endl ; 

    for (G4int i=0;i<daughterCount;i++) 
    {
        const G4VPhysicalVolume* const pv = lv->GetDaughter(i);

        checkSurfTraverse(pv->GetLogicalVolume(),depth+1); 

        // after recursive call

        const G4LogicalBorderSurface* bsurf = GetBorderSurface(pv);
        const G4VPhysicalVolume* bsurf_v1 = bsurf ? bsurf->GetVolume1() : NULL ; 
        const G4VPhysicalVolume* bsurf_v2 = bsurf ? bsurf->GetVolume2() : NULL ; 

        std::cout << " daughter " << i 
                  << " pv " << pv->GetName()
                  << " bsurf " << bsurf 
                  << " bsurf_v1 " << ( bsurf_v1 ? bsurf_v1->GetName() : "-" )
                  << " bsurf_v2 " << ( bsurf_v2 ? bsurf_v2->GetName() : "-" )
                  << std::endl ; 
    }
}





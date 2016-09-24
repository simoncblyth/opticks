#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "PLOG.hh"

#include "CBorderSurfaceTable.hh"
#include "G4LogicalBorderSurface.hh"


CBorderSurfaceTable::CBorderSurfaceTable()
{
    init();
}
   
void CBorderSurfaceTable::init()
{
    int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();

    LOG(info) << "CBorderSurfaceTable::init"
              << " nsurf " << nsurf 
              ;

    const G4LogicalBorderSurfaceTable* bst = G4LogicalBorderSurface::GetSurfaceTable();

    assert( int(bst->size()) == nsurf );

    for(int i=0 ; i < nsurf ; i++)
    {
        G4LogicalBorderSurface* bs = (*bst)[i] ;
        const G4VPhysicalVolume* pv1 = bs->GetVolume1() ;
        const G4VPhysicalVolume* pv2 = bs->GetVolume2() ;

        LOG(info) << std::setw(5) << i 
                  << " pv1 " << pv1->GetName() << " #" << pv1->GetCopyNo() 
                  << " pv2 " << pv2->GetName() << " #" << pv2->GetCopyNo() 
                  ;
    }

}

void CBorderSurfaceTable::dump(const char* msg)
{
    LOG(info) << msg ; 
}


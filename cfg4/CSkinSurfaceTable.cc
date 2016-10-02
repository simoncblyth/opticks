#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "PLOG.hh"

#include "CSkinSurfaceTable.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalSkinSurface.hh"


CSkinSurfaceTable::CSkinSurfaceTable()
{
    init();
}
   
void CSkinSurfaceTable::init()
{
    int nsurf = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();

    LOG(info) << "CSkinSurfaceTable::init"
              << " nsurf " << nsurf 
              ;

    const G4LogicalSkinSurfaceTable* sst = G4LogicalSkinSurface::GetSurfaceTable();

    assert( int(sst->size()) == nsurf );

    for(int i=0 ; i < nsurf ; i++)
    {
        G4LogicalSkinSurface* ss = (*sst)[i] ;
        const G4LogicalVolume* lv = ss->GetLogicalVolume() ;

        std::cout << std::setw(5) << i 
                  << std::setw(35) << ( ss ? ss->GetName() : "NULL" )
                  << " lv " << ( lv ? lv->GetName() : "NULL" ) 
                  << std::endl 
                  ;
    }

}

void CSkinSurfaceTable::dump(const char* msg)
{
    LOG(info) << msg ; 
}


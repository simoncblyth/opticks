#pragma once

#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"


struct U4Traverse
{
    static void Traverse(const G4VPhysicalVolume* const top );
    static void Traverse_r(const G4VPhysicalVolume* const pv, int depth, int sibex,       int& count, int parent_numsib );
    static void Visit(     const G4VPhysicalVolume* const pv, int depth, int sibex, const int& count, int numsib );
};



inline void U4Traverse::Traverse(const G4VPhysicalVolume* const top )
{
    int count = 0 ;
    Traverse_r(top, 0, 0, count, 0 );
}

inline void U4Traverse::Traverse_r(const G4VPhysicalVolume* const pv, int depth, int sibex, int& count, int parent_numsib )
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    int num_child = int(lv->GetNoDaughters()) ;

    Visit(pv, depth, sibex, count, num_child );
    count += 1 ;

    for (int i=0 ; i < num_child ;i++ )
    {
        const G4VPhysicalVolume* const child_pv = lv->GetDaughter(i);

        Traverse_r(child_pv, depth+1, i, count, num_child );
    }
}

inline void U4Traverse::Visit( const G4VPhysicalVolume* const pv, int depth, int sibex, const int& count, int numsib )
{
    const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    assert(placement);
    G4int copy = placement->GetCopyNo() ;

    std::cout
         << " count " << std::setw(6) << count
         << " depth " << std::setw(6) << depth
         << " sibex " << std::setw(6) << sibex
         << " copy  " << std::setw(6) << copy
         << " numsib  " << std::setw(6) << numsib
         << std::endl
         ;
}




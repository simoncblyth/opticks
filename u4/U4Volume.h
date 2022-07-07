#pragma once

#include <string>
class G4VPhysicalVolume ; 

struct U4Volume
{
    static void Traverse( const G4VPhysicalVolume* pv, const char* label ); 
    static void Traverse( const G4LogicalVolume*   lv, const char* label ); 
    static void Traverse_r(const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label ); 
    static std::string DescLV(const G4LogicalVolume* lv); 
};

void U4Volume::Traverse( const G4VPhysicalVolume* pv, const char* label )
{
    Traverse_r( pv, 0, 0, label);
}
void U4Volume::Traverse( const G4LogicalVolume* lv, const char* label  )
{
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), 0, i, label  ); 
}

void U4Volume::Traverse_r( const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    std::cout 
        << " label " << std::setw(10) << label 
        << " depth " << std::setw(2) << depth 
        << " sib " << std::setw(2) << sib 
        << " " << DescLV(lv) 
        << std::endl 
        ; 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), depth+1, i, label ); 
}

std::string U4Volume::DescLV(const G4LogicalVolume* lv)
{
    const G4VSolid* so = lv->GetSolid();
    const G4Material* mt = lv->GetMaterial() ;
    const G4String& mtname = mt->GetName()  ; 
    G4String soname = so->GetName();
    size_t num_daughters = size_t(lv->GetNoDaughters()) ; 

    std::stringstream ss ; 
    ss
        << "U4Volume::DescLV"
        << " num_daughters " << std::setw(4) << num_daughters
        << " mtname " << mtname
        << " soname " << soname
        ;
    std::string s = ss.str(); 
    return s ; 
}


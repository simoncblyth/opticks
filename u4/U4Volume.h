#pragma once

#include <string>
class G4VPhysicalVolume ; 

struct U4Volume
{
    static G4VPhysicalVolume* FindPV( G4VPhysicalVolume* start_pv,  const char* qname );
    static void FindPV_r( G4VPhysicalVolume* pv,  const char* qname, std::vector<G4VPhysicalVolume*>& pvs, int depth ); 

    static void Traverse( const G4VPhysicalVolume* pv, const char* label ); 
    static void Traverse( const G4LogicalVolume*   lv, const char* label ); 
    static void Traverse_r(const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label ); 
    static std::string DescPV(const G4VPhysicalVolume* pv); 
    static std::string DescLV(const G4LogicalVolume* lv); 
};

inline G4VPhysicalVolume*  U4Volume::FindPV( G4VPhysicalVolume* start_pv,  const char* qname ) 
{
    std::vector<G4VPhysicalVolume*> pvs ; 
    FindPV_r(start_pv, qname, pvs, 0 ); 
    return pvs.size() == 1 ? pvs[0] : nullptr ; 
}
inline void U4Volume::FindPV_r( G4VPhysicalVolume* pv,  const char* qname, std::vector<G4VPhysicalVolume*>& pvs, int depth ) 
{
    const G4String& name = pv->GetName(); 
    if( strcmp(name.c_str(), qname) == 0 ) pvs.push_back( pv ); 
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindPV_r( lv->GetDaughter(i), qname, pvs, depth+1 ); 
}




inline void U4Volume::Traverse( const G4VPhysicalVolume* pv, const char* label )
{
    Traverse_r( pv, 0, 0, label);
}
inline void U4Volume::Traverse( const G4LogicalVolume* lv, const char* label  )
{
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), 0, i, label  ); 
}

inline void U4Volume::Traverse_r( const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    std::cout 
        << " label " << std::setw(10) << label 
        << " depth " << std::setw(2) << depth 
        << " sib " << std::setw(2) << sib 
        << " " << DescPV(pv) 
        << " " << DescLV(lv) 
        << std::endl 
        ; 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), depth+1, i, label ); 
}

inline std::string U4Volume::DescPV(const G4VPhysicalVolume* pv)
{
    std::stringstream ss ; 
    ss
        << "U4Volume::DescPV"
        << " pvname " << pv->GetName()   
        ;
    std::string s = ss.str(); 
    return s ; 
}

inline std::string U4Volume::DescLV(const G4LogicalVolume* lv)
{
    const G4VSolid* so = lv->GetSolid();
    const G4Material* mt = lv->GetMaterial() ;
    const G4String& mtname = mt->GetName()  ; 
    G4String soname = so->GetName();
    size_t num_daughters = size_t(lv->GetNoDaughters()) ; 

    std::stringstream ss ; 
    ss
        << "U4Volume::DescLV"
        << " lvname " << lv->GetName()   
        << " num_daughters " << std::setw(4) << num_daughters
        << " mtname " << mtname
        << " soname " << soname
        ;

    std::string s = ss.str(); 
    return s ; 
}


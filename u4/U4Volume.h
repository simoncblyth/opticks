#pragma once

#include "NP.hh"
#include "SStr.hh"
#include <string>
class G4VPhysicalVolume ; 

struct U4Volume
{
    static G4VPhysicalVolume* FindPV( G4VPhysicalVolume* top,  const char* qname );
    static void FindPV_r( G4VPhysicalVolume* pv,  const char* qname, std::vector<G4VPhysicalVolume*>& pvs, int depth ); 


    static G4VPhysicalVolume* FindPVSub( G4VPhysicalVolume* top, const char* sub ); 
    static G4VPhysicalVolume* FindPV_WithSolidName(   G4VPhysicalVolume* top, const char* q_soname, unsigned ordinal, unsigned& count  ) ; 
    static void               FindPV_WithSolidName_r( G4VPhysicalVolume* pv,  const char* q_soname, std::vector<G4VPhysicalVolume*>& pvs, int depth ) ; 


    static void Traverse( const G4VPhysicalVolume* pv, const char* label ); 
    static void Traverse( const G4LogicalVolume*   lv, const char* label ); 
    static void Traverse_r(const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label ); 
    static std::string DescPV(const G4VPhysicalVolume* pv); 
    static std::string DescLV(const G4LogicalVolume* lv); 


    static void GetPV(const G4VPhysicalVolume* top);
    static void GetPV_(const G4VPhysicalVolume*  top,  std::vector<const G4VPhysicalVolume*>& pvs ); 
    static void GetPV_r(const G4VPhysicalVolume* pv,  int depth, std::vector<const G4VPhysicalVolume*>& pvs ); 



    template <typename T> 
    static void WriteStoreNames(const char* dir, const char* name); 

    static void WriteTreeNames( std::vector<const G4VPhysicalVolume*>& pvs , const char* dir, const char* name, const char* opt); 

    static void WriteNames( const G4VPhysicalVolume* top, const char* dir  ); 


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



inline G4VPhysicalVolume* U4Volume::FindPVSub( G4VPhysicalVolume* top, const char* sub )
{
    int spare = -99 ; 
    int ordinal_ = -99 ; 
    const char* q_soname = SStr::ParseStringIntInt(sub, spare, ordinal_ ); 
    assert( spare == 0 && ordinal_ > -1 );  
    unsigned ordinal = ordinal_ ; 
    unsigned count = 0 ; 
    G4VPhysicalVolume* pv_sub = U4Volume::FindPV_WithSolidName( top, q_soname, ordinal, count );  

    LOG(info)
        << " sub " << sub
        << " q_soname " << q_soname
        << " ordinal " << ordinal 
        << " count " << count 
        << " pv_sub " << pv_sub
        << " pv_sub.GetName " << ( pv_sub ? pv_sub->GetName() : "" )
        ;   

    assert( count > 0 && ordinal < count );  
    return pv_sub ; 
}


inline G4VPhysicalVolume* U4Volume::FindPV_WithSolidName( G4VPhysicalVolume* top,  const char* q_soname, unsigned ordinal, unsigned& count  ) 
{
    std::vector<G4VPhysicalVolume*> pvs ; 
    FindPV_WithSolidName_r(top, q_soname, pvs, 0 ); 
    count = pvs.size();   
    return ordinal < pvs.size() ? pvs[ordinal] : nullptr ; 
}
inline void U4Volume::FindPV_WithSolidName_r( G4VPhysicalVolume* pv,  const char* q_soname, std::vector<G4VPhysicalVolume*>& pvs, int depth ) 
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume(); 
    const G4VSolid* so = lv->GetSolid(); 
     
    G4String soname_ = so->GetName() ;  // curiously by value 
    const char* soname = soname_.c_str(); 

    //bool match = strcmp(soname, q_soname) == 0 ; 
    bool match = SStr::StartsWith( soname, q_soname ); 

    if(match) pvs.push_back( pv );

    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindPV_WithSolidName_r( lv->GetDaughter(i), q_soname, pvs, depth+1 ); 
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







inline void U4Volume::GetPV(const G4VPhysicalVolume* top)
{
    std::vector<const G4VPhysicalVolume*> pvs ; 
    GetPV_(top, pvs ); 
}
inline void U4Volume::GetPV_(const G4VPhysicalVolume*  top,  std::vector<const G4VPhysicalVolume*>& pvs )
{
    GetPV_r(top, 0, pvs); 
}
inline void U4Volume::GetPV_r(const G4VPhysicalVolume* pv,  int depth, std::vector<const G4VPhysicalVolume*>& pvs )
{
    pvs.push_back(pv); 
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) GetPV_r( lv->GetDaughter(i), depth+1, pvs ); 
}


template <typename T>
inline void U4Volume::WriteStoreNames(const char* dir, const char* name)
{
    T* store = T::GetInstance() ; 
    std::vector<std::string> names ; 
    typedef typename T::const_iterator IT ; 
    for(IT it=store->begin() ; it != store->end() ; it++) names.push_back((*it)->GetName()) ; 
    NP::WriteNames(dir, name, names); 
}


#include "G4LogicalVolumeStore.hh"
template void U4Volume::WriteStoreNames<G4LogicalVolumeStore>(const char*, const char* ); 

#include "G4PhysicalVolumeStore.hh"
template void U4Volume::WriteStoreNames<G4PhysicalVolumeStore>(const char*, const char* ); 




inline void U4Volume::WriteTreeNames( std::vector<const G4VPhysicalVolume*>& pvs , const char* dir, const char* name, const char* opt)
{
    bool P = strchr(opt, 'P') != nullptr ; 
    bool L = strchr(opt, 'L') != nullptr ; 
    bool S = strchr(opt, 'S') != nullptr ; 

    std::vector<std::string> names ; 
    for(unsigned i=0 ; i < pvs.size() ; i++)
    {
        const G4VPhysicalVolume* pv = pvs[i] ;  
        const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
        const G4VSolid* so = lv->GetSolid(); 
       
        if(P) names.push_back(pv->GetName()); 
        if(L) names.push_back(lv->GetName()); 
        if(S) names.push_back(so->GetName()); 
    }

    LOG(info) << "  names.size " << names.size() << " opt " << opt << " P " << P << " L " << L <<  " S " << S << " name " << name ; 
    NP::WriteNames(dir, name, names); 
}



/**
U4Volume::WriteNames
--------------------------

*WriteStoreNames* just writes the name of each LV and PV once

*WriteTreeNames* repeatedly writes the names of LV, PV and SO 
for every node of the full volume tree.

::

    epsilon:U4VolumeMaker_PVG_WriteNames blyth$ wc -l *.txt
         139 G4LogicalVolumeStore.txt
       51028 G4PhysicalVolumeStore.txt
      336653 L.txt
      336653 P.txt
     1009959 PLS.txt
      336653 S.txt
     2071085 total


**/

inline void U4Volume::WriteNames(const G4VPhysicalVolume* top, const char* dir )
{
    WriteStoreNames<G4LogicalVolumeStore>( dir, "G4LogicalVolumeStore.txt" );  
    WriteStoreNames<G4PhysicalVolumeStore>(dir, "G4PhysicalVolumeStore.txt" );  

    std::vector<const G4VPhysicalVolume*> pvs ; 
    GetPV_(top, pvs ); 
    LOG(info) 
        << " dir " << dir 
        << "  pvs.size " << pvs.size()
        ; 

    WriteTreeNames(pvs, dir, "P.txt", "P" ); 
    WriteTreeNames(pvs, dir, "L.txt", "L" ); 
    WriteTreeNames(pvs, dir, "S.txt", "S" ); 
    WriteTreeNames(pvs, dir, "PLS.txt", "PLS" ); 
}



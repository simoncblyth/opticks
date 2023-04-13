#pragma once

#include <string>
#include <ostream>
#include <sstream>

#include "NP.hh"
#include "sstr.h"


class G4VPhysicalVolume ; 

struct U4Volume
{
    static const G4VPhysicalVolume* FindPV( const G4VPhysicalVolume* top,  const char* qname, int mode=sstr::MATCH_ALL, int maxdeth=-1 );
    static void FindPV_r( const G4VPhysicalVolume* pv,  const char* qname, int mode, std::vector<const G4VPhysicalVolume*>& pvs, int depth, int maxdepth ); 


    static const G4VPhysicalVolume* FindPVSub( const G4VPhysicalVolume* top, const char* sub ); 
    static const G4VPhysicalVolume* FindPV_WithSolidName(   const G4VPhysicalVolume* top, const char* q_soname, unsigned ordinal, unsigned& count  ) ; 
    static void               FindPV_WithSolidName_r( const G4VPhysicalVolume* pv,  const char* q_soname, std::vector<const G4VPhysicalVolume*>& pvs, int depth ) ; 


    static std::string Traverse( const G4VPhysicalVolume* pv, const char* label=nullptr ); 
    static std::string Traverse( const G4LogicalVolume*   lv, const char* label=nullptr ); 
    static void Traverse_r(const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label, std::ostream& ss ); 
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

/**
U4Volume::FindPV
------------------

Find volume named *qname* at or beneath *start_pv*
This is slow when used from stepping around in large geometries. 
*maxdepth* when not -1 restricts the recursion depth to search

**/

inline const G4VPhysicalVolume*  U4Volume::FindPV( const G4VPhysicalVolume* start_pv,  const char* qname, int mode, int maxdepth ) 
{
    std::vector<const G4VPhysicalVolume*> pvs ; 
    FindPV_r(start_pv, qname, mode, pvs, 0, maxdepth ); 
    return pvs.size() == 1 ? pvs[0] : nullptr ; 
}
inline void U4Volume::FindPV_r( const G4VPhysicalVolume* pv,  const char* qname, int mode, std::vector<const G4VPhysicalVolume*>& pvs, int depth, int maxdepth ) 
{
    if(maxdepth > -1 && depth > maxdepth) return ;   
    const G4String& name_ = pv->GetName(); 
    const char* name = name_.c_str() ; 
    if(sstr::Match_(name, qname, mode)) pvs.push_back( pv ); 
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindPV_r( lv->GetDaughter(i), qname, mode, pvs, depth+1, maxdepth ); 
}





inline const G4VPhysicalVolume* U4Volume::FindPVSub( const G4VPhysicalVolume* top, const char* sub )
{
    int spare = -99 ; 
    int ordinal_ = -99 ; 
    const char* q_soname = sstr::ParseStringIntInt(sub, spare, ordinal_ ); 
    assert( spare == 0 && ordinal_ > -1 );  
    unsigned ordinal = ordinal_ ; 
    unsigned count = 0 ; 
    const G4VPhysicalVolume* pv_sub = U4Volume::FindPV_WithSolidName( top, q_soname, ordinal, count );  

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


inline const G4VPhysicalVolume* U4Volume::FindPV_WithSolidName( const G4VPhysicalVolume* top,  const char* q_soname, unsigned ordinal, unsigned& count  ) 
{
    std::vector<const G4VPhysicalVolume*> pvs ; 
    FindPV_WithSolidName_r(top, q_soname, pvs, 0 ); 
    count = pvs.size();   
    return ordinal < pvs.size() ? pvs[ordinal] : nullptr ; 
}
inline void U4Volume::FindPV_WithSolidName_r( const G4VPhysicalVolume* pv,  const char* q_soname, std::vector<const G4VPhysicalVolume*>& pvs, int depth ) 
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume(); 
    const G4VSolid* so = lv->GetSolid(); 
     
    G4String soname_ = so->GetName() ;  // curiously by value 
    const char* soname = soname_.c_str(); 

    bool match = sstr::MatchStart( soname, q_soname ); 

    if(match) pvs.push_back( pv );

    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindPV_WithSolidName_r( lv->GetDaughter(i), q_soname, pvs, depth+1 ); 
}
















inline std::string U4Volume::Traverse( const G4VPhysicalVolume* pv, const char* label )
{
    std::stringstream ss ; 
    Traverse_r( pv, 0, 0, label, ss );
    std::string s = ss.str(); 
    return s ; 
}

inline std::string U4Volume::Traverse( const G4LogicalVolume* lv, const char* label )
{
    std::stringstream ss ; 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), 0, i, label, ss  ); 
    std::string s = ss.str(); 
    return s ; 
}

inline void U4Volume::Traverse_r( const G4VPhysicalVolume* pv, int depth, size_t sib, const char* label, std::ostream& ss )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;

    if(label) ss << " label " << std::setw(10) << label ; 
    ss
        << " dep " << std::setw(2) << depth 
        << " sib " << std::setw(2) << sib 
        << " " << DescPV(pv) 
        << " " << DescLV(lv) 
        << std::endl 
        ; 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) Traverse_r( lv->GetDaughter(i), depth+1, i, label, ss ); 
}


inline std::string U4Volume::DescPV(const G4VPhysicalVolume* pv)
{
    std::stringstream ss ; 
    ss
        << "DescPV "
        << std::setw(20) << pv->GetName()   
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
        << "DescLV "
        << std::setw(20) << lv->GetName()   
        << " nd " << std::setw(4) << num_daughters
        << " mt " << std::setw(8) << mtname
        << " so " << soname
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



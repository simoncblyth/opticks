#pragma once
/**
U4Touchable.h
================

The touchable history is from inside to outside, so "deeper" corresponds to wider volumes.

A version of this is also available within the monolith at 

    $JUNOTOP/junosw/Simulation/DetSimV2/SimUtil/SimUtil/S4Touchable.h 

**/

#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4TouchableHistory.hh"
#include "G4VSolid.hh"

struct U4Touchable
{
    enum { MATCH_ALL, MATCH_START, MATCH_END } ; 
    static bool Match(      const char* s, const char* q, int mode) ; 
    static bool MatchAll(   const char* s, const char* q) ; 
    static bool MatchStart( const char* s, const char* q) ; 
    static bool MatchEnd(   const char* s, const char* q) ; 

    static const G4VPhysicalVolume* FindPV( const G4VTouchable* touch, const char* qname, int mode=MATCH_ALL ); 
    static int ReplicaNumber(const G4VTouchable* touch, const char* replica_name_select ) ; 
    static int ReplicaDepth(const G4VTouchable* touch, const char* replica_name_select ) ; 
    static int TouchDepth(const G4VTouchable* touch ); 
    static bool HasMoreThanOneDaughterWithName( const G4LogicalVolume* lv, const char* name); 

    static std::string Brief(const G4VTouchable* touch ); 
    static std::string Desc(const G4VTouchable* touch, int so_wid=20, int pv_wid=20);
};


inline bool U4Touchable::Match( const char* s, const char* q, int mode )
{
    bool ret = false ; 
    switch(mode)
    {   
        case MATCH_ALL:    ret = MatchAll(  s, q) ; break ; 
        case MATCH_START:  ret = MatchStart(s, q) ; break ; 
        case MATCH_END:    ret = MatchEnd(  s, q) ; break ; 
    }   
    return ret ;
}

inline bool U4Touchable::MatchAll( const char* s, const char* q)
{
    return s && q && strcmp(s, q) == 0 ; 
}
inline bool U4Touchable::MatchStart( const char* s, const char* q)
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ; 
}
inline bool U4Touchable::MatchEnd( const char* s, const char* q)
{
    int pos = strlen(s) - strlen(q) ;
    return pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 ; 
}

/**
U4Touchable::FindPV
---------------------

Find a PV by name in the touch stack, this is much quicker than the recursive U4Volume::FindPV 

**/

inline const G4VPhysicalVolume* U4Touchable::FindPV( const G4VTouchable* touch, const char* qname, int mode )
{
    int nd = touch->GetHistoryDepth();
    int count = 0 ; 
    const G4VPhysicalVolume* qpv = nullptr ;  
    for (int d=0 ; d < nd ; ++d ) 
    {   
        const G4VPhysicalVolume* dpv = touch->GetVolume(d);
        const char* dpv_name = dpv->GetName().c_str() ;
        if(Match(dpv_name, qname, mode))
        {
            qpv = dpv ; 
            count += 1 ;
        }
    }
    return qpv ; 
} 



inline int U4Touchable::ReplicaNumber(const G4VTouchable* touch, const char* replica_name_select )  // static 
{
    int d = ReplicaDepth(touch, replica_name_select);
    bool found = d > -1 ; 
    int repno = found ? touch->GetReplicaNumber(d) : d  ;

#ifdef U4TOUCHABLE_DEBUG
    if(found) std::cerr 
        << "U4Touchable::ReplicaNumber"
        << " found " << found
        << " repno " << repno
        << std::endl 
        ;  
#endif

    return repno ;
}

/**
U4Touchable::ReplicaDepth
---------------------------

Starting from touch depth look outwards at (volume, mother_volume) 
pairs checking for a depth at which the mother_volume has more than one 
daughter with the same name as the volume. 

This means the volume has
at least one same named sibling making this the replica depth. 
When no such depth is found return -1. 

For non-null replica_name_select the name search is restricted to names 
of logical volumes that contain the replica_name_select string. 
Depending on the the names of the replica logical volumes 
a suitable "replica_name_select" string (eg "PMT") 
may dramatically speedup the search as pointless searches 
over thousands of volumes are avoided. Of course this 
depends on suitable naming of replica logical volumes. 

**/

inline int U4Touchable::ReplicaDepth(const G4VTouchable* touch, const char* replica_name_select )   // static
{
    int nd = touch->GetHistoryDepth();
    int t = TouchDepth(touch); 
    bool expected = t > -1 && t < nd ; 
   
    /* 
    if(!expected) std::cerr 
        << "U4Touchable::ReplicaDepth"
        << " UNEXPECTED "
        << " t " << t 
        << " nd " << nd
        << std::endl
        ; 

    assert(expected); 
    */

    if(!expected) return -2 ; 

    for (int d=t ; d < nd-1; ++d ) 
    {   
        const G4VPhysicalVolume* dpv = touch->GetVolume(d);
        const G4VPhysicalVolume* mpv = touch->GetVolume(d+1);

        const G4LogicalVolume* dlv = dpv->GetLogicalVolume();
        const G4LogicalVolume* mlv = mpv->GetLogicalVolume();

        //const G4VSensitiveDetector* dsd = dlv->GetSensitiveDetector(); 
        //const G4VSensitiveDetector* msd = mlv->GetSensitiveDetector(); 
        //bool sd_skip = dsd == nullptr && msd == nullptr ; 

        bool hierarchy = dpv->GetMotherLogical() == mlv ; 
        assert(hierarchy); 

        const char* dlv_name = dlv->GetName().c_str() ; 
        bool name_skip = replica_name_select && strstr(dlv_name, replica_name_select) == nullptr ; 
        //bool skip = name_skip || sd_skip ; 

#ifdef U4TOUCHABLE_DEBUG
        std::cerr 
            << "U4Touchable::ReplicaDepth"
            << " d " << d 
            << " nd " << nd 
            << " dlv_name " << dlv_name
            << " replica_name_select " << ( replica_name_select ? replica_name_select : "-" )
            << " name_skip " << name_skip
            << " skip " << skip
            << std::endl 
            ; 
#endif

        // skip:true when replica_name_select is provided but the string is not found within the dlv name 
        // HMM: thats a negative way of doing things, positive approach would be more restrictive so faster
        if(name_skip) continue ; 
        bool found = HasMoreThanOneDaughterWithName(mlv, dlv_name) ;  

#ifdef U4TOUCHABLE_DEBUG
        if(found)std::cerr 
            << "U4Touchable::ReplicaDepth"
            << " d " << d 
            << " dlv_name " << dlv_name
            << " found " << found
            << std::endl 
            ;
#endif

        if(found) return d ; 
    }
    return -1 ;
}

/**
U4Touchable::TouchDepth
-------------------------

Depth of touch volume, -1 if not found. 

**/

inline int U4Touchable::TouchDepth(const G4VTouchable* touch ) // static
{
    const G4VPhysicalVolume* tpv = touch->GetVolume() ;
    int t = -1 ; 
    for(int i=0 ; i < touch->GetHistoryDepth() ; i++) 
    {   
        const G4VPhysicalVolume* ipv = touch->GetVolume(i) ; 
        if(ipv == tpv) 
        {
            t = i ;  
            break ; 
        }
    } 
    return t ; 
}


/**
U4Touchable::HasMoreThanOneDaughterWithName
---------------------------------------------

When called with name lAcrylic which has num_dau 46276
this is real expensive. 

**/

inline bool U4Touchable::HasMoreThanOneDaughterWithName( const G4LogicalVolume* lv, const char* name)  // static
{
    int num_dau = lv->GetNoDaughters();
    if(num_dau <= 1) return false ; 

#ifdef U4TOUCHABLE_DEBUG
    bool heavy = num_dau > 45000 ;
    if(heavy) std::cerr 
        << "U4Touchable::HasMoreThanOneDaughterWithName"
        << " num_dau " << num_dau 
        << " name " << name
        << " lv.name " << lv->GetName() 
        << std::endl 
        ;
#endif


    int count = 0;  
    for (int k=0; k < num_dau ; ++k)   
    {
        const G4VPhysicalVolume* kpv = lv->GetDaughter(k) ;
        const G4LogicalVolume*   klv = kpv->GetLogicalVolume() ;
        const char* klv_name = klv->GetName().c_str() ;

#ifdef U4TOUCHABLE_DEBUG
        if(heavy) std::cerr 
           << "U4Touchable::HasMoreThanOneDaughterWithName"
           << " k " << k 
           << " kpv.name " << kpv->GetName()
           << " klv_name " << klv_name
           << " count " << count 
           << std::endl
           ; 
#endif

        if(strcmp(name, klv_name)==0) count += 1 ;
        if(count > 1) return true ;
    }
    return false ; 
}




inline std::string U4Touchable::Brief(const G4VTouchable* touch )
{
    std::stringstream ss ; 
    ss << "U4Touchable::Brief"
       << " HistoryDepth " << std::setw(2) <<  touch->GetHistoryDepth()
       << " TouchDepth " << std::setw(2) << TouchDepth(touch)
       << " ReplicaDepth " << std::setw(2) << ReplicaDepth(touch, nullptr)
       << " ReplicaNumber " << std::setw(6) << ReplicaNumber(touch, nullptr)
       ; 
    return ss.str(); 
}
inline std::string U4Touchable::Desc(const G4VTouchable* touch, int so_wid, int pv_wid )
{
    int history_depth = touch->GetHistoryDepth();
    int touch_depth = TouchDepth(touch); 
    int replica_depth = ReplicaDepth(touch, nullptr); 
    int replica_number = ReplicaNumber(touch, nullptr); 

    std::stringstream ss ; 
    ss << "U4Touchable::Desc"
       << " HistoryDepth " << std::setw(2) << history_depth 
       << " TouchDepth " << std::setw(2) << touch_depth 
       << " ReplicaDepth " << std::setw(2) << replica_depth
       << " ReplicaNumber " << std::setw(6) << replica_number 
       << std::endl 
       ; 

    for(int i=0 ; i< history_depth ; i++) 
    { 
        G4VPhysicalVolume* pv = touch->GetVolume(i); 
        G4LogicalVolume* lv = pv->GetLogicalVolume();
        G4int nd = lv->GetNoDaughters();
        G4VSolid* so = touch->GetSolid(i); 
        G4int cp = touch->GetReplicaNumber(i); 

        ss << " i " << std::setw(2) << i 
           << " cp " << std::setw(6)  << cp
           << " nd " << std::setw(6)  << nd
           << " so " << std::setw(so_wid) << so->GetName()
           << " pv " << std::setw(pv_wid) << pv->GetName()
           << std::endl ; 
    }   
    std::string s = ss.str(); 
    return s ; 
}



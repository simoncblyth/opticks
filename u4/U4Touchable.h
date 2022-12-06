#pragma once
/**
U4Touchable.h
================

::

    U4Recorder::UserSteppingAction_Optical@423: U4Touchable::Desc depth 4
     i  0 cp      0 nd      7 so   hama_inner2_solid_1_4 pv  hama_inner2_phys
     i  1 cp      0 nd      2 so     hama_body_solid_1_4 pv    hama_body_phys
     i  2 cp      0 nd      1 so      hama_pmt_solid_1_4 pv       hama_log_pv
     i  3 cp      0 nd      1 so             Water_solid pv       Water_lv_pv

**/

#include <string>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4TouchableHistory.hh"
#include "G4VSolid.hh"

struct U4Touchable
{
    static int GetReplicaDepth(const G4VTouchable* touch) ; 
    static int GetReplicaNumber(const G4VTouchable* touch) ; 
    static std::string Desc(const G4VTouchable* touch);
};

inline int U4Touchable::GetReplicaDepth(const G4VTouchable* touch) 
{
    const G4VPhysicalVolume* tpv = touch->GetVolume() ;
    int nd = touch->GetHistoryDepth();

    for(int i=0; i<nd; i++) 
    {   
        const G4VPhysicalVolume* ipv = touch->GetVolume(i) ; 
        if(ipv == tpv) 
        {   
            int j=1;
            for (j=1; j < (nd-i); ++j) 
            {   
                int d = i+j-1 ; 

                const G4VPhysicalVolume* dpv = touch->GetVolume(d);
                const G4LogicalVolume* dlv = dpv->GetLogicalVolume();
                const G4String& dlv_name = dlv->GetName() ; 

                const G4VPhysicalVolume* jpv = touch->GetVolume(i+j);
                const G4LogicalVolume* jlv = jpv->GetLogicalVolume();
                int jlv_dau = jlv->GetNoDaughters();

                if (jlv_dau > 1)  
                {   
                    int count = 0;  
                    // looks like the count < 2 will never do anything due to below break 
                    for (int k=0; (count<2) && (k < jlv_dau); ++k)   
                    {
                        const G4VPhysicalVolume* kpv = jlv->GetDaughter(k) ;
                        const G4LogicalVolume*   klv = kpv->GetLogicalVolume() ;
                        const G4String& klv_name = klv->GetName() ;
                        if (dlv_name == klv_name) ++count ;
                    }
                    if(count > 1) return d ;
                }
            }
        }
    }
    return -1 ;
}


inline int U4Touchable::GetReplicaNumber(const G4VTouchable* touch)  // static 
{
    int d = GetReplicaDepth(touch);
    return d > 0 ? touch->GetReplicaNumber(d) : -1  ;
}



inline std::string U4Touchable::Desc(const G4VTouchable* touch)
{
    int depth = touch->GetHistoryDepth();
    int replica_depth = GetReplicaDepth(touch); 
    int replica_number = GetReplicaNumber(touch); 

    std::stringstream ss ; 
    ss << "U4Touchable::Desc" << std::endl 
       << " touch->GetHistoryDepth  " << depth << std::endl 
       << " replica_depth " << replica_depth << std::endl 
       << " replica_number " << replica_number << std::endl 
       ; 

    for(int i=0 ; i<depth ; i++) 
    { 
        G4VPhysicalVolume* pv = touch->GetVolume(i); 
        G4LogicalVolume* lv = pv->GetLogicalVolume();
        G4int nd = lv->GetNoDaughters();
        G4VSolid* so = touch->GetSolid(i); 
        G4int cp = touch->GetReplicaNumber(i); 

        ss << " i " << std::setw(2) << i 
           << " cp " << std::setw(6)  << cp
           << " nd " << std::setw(6)  << nd
           << " so " << std::setw(40) << so->GetName()
           << " pv " << std::setw(60) << pv->GetName()
           << std::endl ; 
    }   
    std::string s = ss.str(); 
    return s ; 
}



#pragma once

#include "stree.h"
#include "U4Tree.h"
#include "G4VSolid.hh"

struct U4Simtrace
{
    stree st ; 
    U4Tree ut ; 

    U4Simtrace(const G4VPhysicalVolume* pv ); 
    void scan(const char* base ); 
}; 

inline U4Simtrace::U4Simtrace(const G4VPhysicalVolume* pv )
    :
    ut(&st, pv)
{
}

/**
U4Simtrace::scan
-------------------

HMM: could use NPFold instead of saving to file after every scan ?
(actually the simple approach probably better in terms of memory)
**/

inline void U4Simtrace::scan(const char* base)
{
    st.save_trs(base); 
    assert( st.soname.size() == ut.solids.size() ); 
    for(unsigned i=0 ; i < st.soname.size() ; i++)  // over unique solid names
    {
        const char* soname = st.soname[i].c_str(); 
        const G4VSolid* solid = ut.solids[i] ; 
        G4String name = solid->GetName(); 
        assert( strcmp( name.c_str(), soname ) == 0 ); 
        SSimtrace::Scan(solid, base) ;   
    }
}



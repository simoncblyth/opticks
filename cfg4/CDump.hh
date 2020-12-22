#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4MaterialPropertiesTable ; 

class CFG4_API CDump {
    public:
        static const unsigned EDGEITEMS ; 
        static void G4MaterialPropertiesTable_(const char* name, const G4MaterialPropertiesTable* mpt); 
        static void G4LogicalBorderSurfaceTable_();  
        static void G4LogicalSkinSurface_();  
        static void G4MaterialTable_(); 
        static void G4(const char* cfg);  
        static void G4Version_(); 

};

#include "CFG4_TAIL.hh"




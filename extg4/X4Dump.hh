#pragma once

#include "X4_API_EXPORT.hh"

class G4MaterialPropertiesTable ; 

class X4_API X4Dump {
    public:
        static const unsigned EDGEITEMS ; 
        static void G4MaterialPropertiesTable_(const char* name, const G4MaterialPropertiesTable* mpt); 
        static void G4LogicalBorderSurfaceTable_();  
        static void G4LogicalSkinSurface_();  
        static void G4MaterialTable_(); 
        static void G4(const char* cfg);  
        static void G4Version_(); 

};





#pragma once

#include "X4_API_EXPORT.hh"

class GSurfaceLib ; 

#include "G4LogicalSkinSurface.hh"   // forced to include for the typedef 
#include "X4_API_EXPORT.hh"

/**
X4LogicalSkinSurfaceTable
============================

**/

class X4_API X4LogicalSkinSurfaceTable 
{
    public:
        static void Convert(GSurfaceLib* dst);
    private:
        X4LogicalSkinSurfaceTable(GSurfaceLib* dst);
        void init();
    private:
        const G4LogicalSkinSurfaceTable*  m_src ; 
        GSurfaceLib*                        m_dst ; 
        

};

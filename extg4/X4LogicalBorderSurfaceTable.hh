#pragma once

#include "X4_API_EXPORT.hh"

class GSurfaceLib ; 

#include "G4LogicalBorderSurface.hh"   // forced to include for the typedef 
#include "X4_API_EXPORT.hh"

/**
X4LogicalBorderSurfaceTable
============================

**/

class X4_API X4LogicalBorderSurfaceTable 
{
    public:
        static void Convert(GSurfaceLib* slib);
    private:
        X4LogicalBorderSurfaceTable(GSurfaceLib* slib);
        void init();
    private:
        const G4LogicalBorderSurfaceTable*  m_table ; 
        GSurfaceLib*                        m_slib ; 
        

};

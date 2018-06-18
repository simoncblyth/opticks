#pragma once

class G4LogicalBorderSurface ;
class GBorderSurface ; 

#include "X4_API_EXPORT.hh"

/**
X4LogicalBorderSurface
=======================

**/

class X4_API X4LogicalBorderSurface
{
    public:
        static GBorderSurface* Convert(const G4LogicalBorderSurface* lbs);
    private:
        X4LogicalBorderSurface(const G4LogicalBorderSurface* lbs);
        void init();
        GBorderSurface* getBorderSurface() const ; 

    private:
        const G4LogicalBorderSurface*  m_lbs ; 
        GBorderSurface*                m_bs ;  
};



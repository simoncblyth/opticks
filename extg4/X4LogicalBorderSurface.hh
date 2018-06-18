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
        static GBorderSurface* Convert(const G4LogicalBorderSurface* src);
        static int GetItemIndex( const G4LogicalBorderSurface* item ) ;
};



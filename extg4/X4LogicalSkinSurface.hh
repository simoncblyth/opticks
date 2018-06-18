#pragma once

class G4LogicalSkinSurface ;
class GSkinSurface ; 

#include "X4_API_EXPORT.hh"

/**
X4LogicalSkinSurface
=======================

**/

class X4_API X4LogicalSkinSurface
{
    public:
        static GSkinSurface* Convert(const G4LogicalSkinSurface* src);
        static int GetItemIndex( const G4LogicalSkinSurface* src ) ;
};



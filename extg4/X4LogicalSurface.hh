#pragma once


#include "X4_API_EXPORT.hh"

/**
X4LogicalSurface
===================

**/

class G4LogicalSurface ;
template <typename T> class GPropertyMap ; 

class X4_API X4LogicalSurface
{
    public:
        static void Convert(GPropertyMap<float>* dst,  const G4LogicalSurface* src);
};



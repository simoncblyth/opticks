#pragma once

#include "G4RotationMatrix.hh"
#include "X4_API_EXPORT.hh"

// subclass to use protected ctor

template <typename T>
struct X4_API X4RotationMatrix  : public G4RotationMatrix 
{
     X4RotationMatrix( 
           T xx, T xy, T xz,
           T yx, T yy, T yz, 
           T zx, T zy, T zz 
     )
     : 
       G4RotationMatrix( xx, xy, xz ,   
                         yx, yy, yz ,
                         zx, zy, zz )
    {}

};


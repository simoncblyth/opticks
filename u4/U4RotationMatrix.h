#pragma once
/**
U4RotationMatrix.h
====================

typedef CLHEP::HepRotation G4RotationMatrix;

g4-cls Rotation


Subclass in order to allow use of protected ctor


**/

#include "G4RotationMatrix.hh"
#include <cstring>


struct U4RotationMatrix : public G4RotationMatrix
{
    enum {
       X = 0x1 << 0 ,
       Y = 0x1 << 1 , 
       Z = 0x1 << 2  
    }; 

     static U4RotationMatrix* Flip(unsigned mask) ;
     static U4RotationMatrix* Flip(const char* axes) ;
     static unsigned FlipMask(const char* axes); 

     U4RotationMatrix( 
           double xx, double xy, double xz, 
           double yx, double yy, double yz, 
           double zx, double zy, double zz  
     ); 
};

inline U4RotationMatrix::U4RotationMatrix( 
           double xx, double xy, double xz, 
           double yx, double yy, double yz, 
           double zx, double zy, double zz  
     )   
     :   
       G4RotationMatrix( xx, xy, xz ,   
                         yx, yy, yz ,
                         zx, zy, zz )
{}  


inline U4RotationMatrix* U4RotationMatrix::Flip(unsigned mask )
{
    double XX = mask & X ? -1. : 1. ; 
    double YY = mask & Y ? -1. : 1. ; 
    double ZZ = mask & Z ? -1. : 1. ; 

    return new U4RotationMatrix( XX, 0., 0., 
                                 0., YY, 0., 
                                 0., 0., ZZ ); 
}
inline unsigned U4RotationMatrix::FlipMask(const char* axes)
{
    unsigned mask = 0 ; 
    if(axes && strstr(axes, "X")) mask |= X ; 
    if(axes && strstr(axes, "Y")) mask |= Y ; 
    if(axes && strstr(axes, "Z")) mask |= Z ; 
    return mask ; 
}
inline U4RotationMatrix* U4RotationMatrix::Flip(const char* axes)
{
    return Flip(FlipMask(axes)); 
}





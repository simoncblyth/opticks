#pragma once
/**
U4RotationMatrix.h
====================

typedef CLHEP::HepRotation G4RotationMatrix;

g4-cls Rotation


Subclass in order to allow use of protected ctor


**/

#include "G4RotationMatrix.hh"
#include <iomanip>
#include <cstring>


struct U4RotationMatrix : public G4RotationMatrix
{
    enum {
       X = 0x1 << 0 ,
       Y = 0x1 << 1 , 
       Z = 0x1 << 2  
    }; 

     static std::string Desc(const G4RotationMatrix* rot ); 

     static U4RotationMatrix* Flip(unsigned mask) ;
     static U4RotationMatrix* Flip(const char* axes) ;
     static unsigned FlipMask(const char* axes); 

     U4RotationMatrix(const double* src, bool flip); 

     U4RotationMatrix( 
           double xx, double xy, double xz, 
           double yx, double yy, double yz, 
           double zx, double zy, double zz  
     ); 
};

inline std::string U4RotationMatrix::Desc(const G4RotationMatrix* rot )
{
    int wid = 10 ; 
    int prc = 3 ; 

    std::stringstream ss ; 

    ss << "U4RotationMatrix::Desc" 
       << ( rot ? "" : " null" ) 
       << std::endl 
       ;

    if(rot) ss
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->xx() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->xy() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->xz()
        << std::endl 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->yx() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->yy() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->yz()
        << std::endl 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->zx() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->zy() 
        << " " << std::fixed << std::setw(wid) << std::setprecision(prc) << rot->zz()
        << std::endl 
        ;

    std::string s = ss.str(); 
    return s ; 
}


/**

The input array a16 is assumed to be of 16 elements 
The array element to rotation matrix element mapping depends on f.

For f:false:: 
 
    xx:00  xy:01  xz:02   -- 
    yx:04  yy:05  yz:06   -- 
    zx:08  zy:09  zz:10   --
       --     --     --   --

For f:true:: 

    xx:00  xy:04  xz:08   -- 
    yx:01  yy:05  yz:09   -- 
    zx:02  zy:06  zz:10   --
       --     --     --   --

**/

inline U4RotationMatrix::U4RotationMatrix(const double* a16, bool f )
      :    
         G4RotationMatrix( a16[0],     a16[f?4:1], a16[f?8:2], 
                           a16[f?1:4], a16[5],     a16[f?9:6],
                           a16[f?2:8], a16[f?6:9], a16[10]     )
{}


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





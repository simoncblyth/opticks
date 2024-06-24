#pragma once
/**
sgeomtools.h
=============

Adapt some extracts from G4GeomTools

**/

#include <cmath>
#include "scuda.h"

struct sgeomtools
{
    static constexpr const double kCarTolerance = 1e-5 ; 

    static void Set( double2& p, double x, double y );

    static bool DiskExtent( 
            double rmin, 
            double rmax,
            double startPhi,
            double deltaPhi,
            double2& pmin,
            double2& pmax); 

    static void DiskExtent( 
            double rmin, 
            double rmax,
            double sinPhiStart,
            double cosPhiStart,
            double sinPhiEnd,
            double cosPhiEnd,
            double2& pmin,
            double2& pmax); 
};


inline void sgeomtools::Set( double2& p, double x, double y )
{
    p.x = x ; 
    p.y = y ; 
} 

inline bool sgeomtools::DiskExtent( 
            double rmin, 
            double rmax,
            double startPhi,
            double deltaPhi,
            double2& pmin,
            double2& pmax)   // static
{
    Set(pmin,0.,0.); 
    Set(pmax,0.,0.); 

    if(rmin < 0.)    return false ; 
    if(rmax <= rmin) return false ; 
    if(deltaPhi <= 0.) return false ; 

    Set(pmin,-rmax,-rmax); 
    Set(pmax, rmax, rmax); 

    if(deltaPhi >= M_PI*2. ) return true ; 
    double endPhi = startPhi + deltaPhi ; 

    DiskExtent( rmin,
                rmax,
                std::sin(startPhi),
                std::cos(startPhi),
                std::sin(endPhi),
                std::cos(endPhi),
                pmin,
                pmax); 

    return true ;   
}


inline void sgeomtools::DiskExtent( 
            double rmin, 
            double rmax,
            double sinStart,
            double cosStart,
            double sinEnd,
            double cosEnd,
            double2& pmin,
            double2& pmax)   // static
{
    Set(pmin,-rmax,-rmax); 
    Set(pmax, rmax, rmax); 

  if (std::abs(sinEnd-sinStart) < kCarTolerance && 
      std::abs(cosEnd-cosStart) < kCarTolerance) return;

  // get start and end quadrants
  //
  //      1 | 0
  //     ---+--- 
  //      3 | 2
  //
  int icase = (cosEnd < 0) ? 1 : 0;
  if (sinEnd   < 0) icase += 2;
  if (cosStart < 0) icase += 4;
  if (sinStart < 0) icase += 8;

  switch (icase)
  {
  // start quadrant 0
  case  0:                                 // start->end : 0->0
    if (sinEnd < sinStart) break;
    Set(pmin,rmin*cosEnd,rmin*sinStart);
    Set(pmax,rmax*cosStart,rmax*sinEnd  );
    break;
  case  1:                                 // start->end : 0->1
    Set(pmin,rmax*cosEnd,std::min(rmin*sinStart,rmin*sinEnd));
    Set(pmax,rmax*cosStart,rmax  );
    break;
  case  2:                                 // start->end : 0->2
    Set(pmin,-rmax,-rmax);
    Set(pmax,std::max(rmax*cosStart,rmax*cosEnd),rmax);
    break;
  case  3:                                 // start->end : 0->3
    Set(pmin,-rmax,rmax*sinEnd);
    Set(pmax,rmax*cosStart,rmax);
    break;
  // start quadrant 1
  case  4:                                 // start->end : 1->0
    Set(pmin,-rmax,-rmax);
    Set(pmax,rmax,std::max(rmax*sinStart,rmax*sinEnd));
    break;
  case  5:                                 // start->end : 1->1
    if (sinEnd > sinStart) break;
    Set(pmin,rmax*cosEnd,rmin*sinEnd  );
    Set(pmax,rmin*cosStart,rmax*sinStart);
    break;
  case  6:                                 // start->end : 1->2
    Set(pmin,-rmax,-rmax);
    Set(pmax,rmax*cosEnd,rmax*sinStart);
    break;
  case  7:                                 // start->end : 1->3
    Set(pmin,-rmax,rmax*sinEnd);
    Set(pmax,std::max(rmin*cosStart,rmin*cosEnd),rmax*sinStart);
    break;
  // start quadrant 2
  case  8:                                 // start->end : 2->0
    Set(pmin,std::min(rmin*cosStart,rmin*cosEnd),rmax*sinStart);
    Set(pmax,rmax,rmax*sinEnd);
    break;
  case  9:                                 // start->end : 2->1
    Set(pmin,rmax*cosEnd,rmax*sinStart);
    Set(pmax,rmax,rmax);
    break;
  case 10:                                 // start->end : 2->2
    if (sinEnd < sinStart) break;
    Set(pmin,rmin*cosStart,rmax*sinStart);
    Set(pmax,rmax*cosEnd,rmin*sinEnd  );
    break;
  case 11:                                 // start->end : 2->3
    Set(pmin,-rmax,std::min(rmax*sinStart,rmax*sinEnd));
    Set(pmax,rmax,rmax);
    break;
  // start quadrant 3
  case 12:                                 // start->end : 3->0
    Set(pmin,rmax*cosStart,-rmax);
    Set(pmax,rmax,rmax*sinEnd);
    break;
  case 13:                                 // start->end : 3->1
    Set(pmin,std::min(rmax*cosStart,rmax*cosEnd),-rmax);
    Set(pmax,rmax,rmax);
    break;
  case 14:                                 // start->end : 3->2
    Set(pmin,rmax*cosStart,-rmax);
    Set(pmax,rmax*cosEnd,std::max(rmin*sinStart,rmin*sinEnd));
    break;
  case 15:                                 // start->end : 3->3
    if (sinEnd > sinStart) break;
    Set(pmin,rmax*cosStart,rmax*sinEnd);
    Set(pmax,rmin*cosEnd,rmin*sinStart);
    break;
  }
  return;
}

 

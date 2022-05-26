#pragma once

#include <vector>
struct quad6 ; 
struct NP ; 

#include "sevent.h"  // just enum { XYZ, YZ, XZ, XY } ;   TODO: eliminate 

struct SGenstep
{
    static const char* XYZ_ ; 
    static const char* YZ_  ; 
    static const char* XZ_  ; 
    static const char* XY_  ; 
    static const char* GridAxesName( int gridaxes ); 
    static int GridAxes(int nx, int ny, int nz); 
    static unsigned GenstepID( int ix, int iy, int iz, int iw=0 ) ; 

    static void ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep ); 

    static NP* MakeArray(const std::vector<quad6>& gs ); 

}; 

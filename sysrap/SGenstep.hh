#pragma once

#include <string>
#include <vector>
struct quad6 ; 
struct NP ; 

#include "sxyz.h" 
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SGenstep
{
    static const char* XYZ_ ; 
    static const char* YZ_  ; 
    static const char* XZ_  ; 
    static const char* XY_  ; 
    static const char* GridAxesName( int gridaxes ); 
    static int GridAxes(int nx, int ny, int nz); 
    static unsigned GenstepID( int ix, int iy, int iz, int iw=0 ) ; 

    static void ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep ); 
    static int GetGencode( const quad6& gs ) ; 
    static int GetNumPhoton( const quad6& gs ) ; 

    static const quad6& GetGenstep(const NP* gs, unsigned gs_idx ); 
    static int GetGencode(    const NP* gs, unsigned gs_idx  ); 
    static int GetNumPhoton( const NP* gs, unsigned gs_idx  ); 

    static void Check(const NP* gs); 
    static NP* MakeArray(const std::vector<quad6>& gs ); 

    static std::string Desc(const NP* gs, int edgeitems); 

}; 

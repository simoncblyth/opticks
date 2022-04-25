#pragma once

struct NP ; 
struct quad6 ; 
struct torch ; 
struct uint4 ; 
template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

#include "sevent.h"  // just enum { XYZ, YZ, XZ, XY } ;   TODO: eliminate 

struct SYSRAP_API SEvent
{
    static const plog::Severity LEVEL ; 


    static const char* XYZ_ ; 
    static const char* YZ_  ; 
    static const char* XZ_  ; 
    static const char* XY_  ; 
    static const char* GridAxesName( int gridaxes ); 
    static int GridAxes(int nx, int ny, int nz); 


    static NP* MakeDemoGensteps(const char* config=nullptr);  


    static void FillTorchGenstep( torch& gs, unsigned genstep_id, unsigned numphoton_per_genstep ); 
    static NP* MakeTorchGensteps(const char* config=nullptr);  


    static NP* MakeGensteps(const std::vector<quad6>& gs ); 
    static void StandardizeCEGS(        const float4& ce,       std::vector<int>& cegs, float gridscale );
    static void GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, bool ce_offset ) ; 


    static void ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep ); 

    static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, bool ce_offset, bool ce_scale ) ;
    static NP* MakeCountGensteps(const char* config=nullptr);
    static unsigned SumCounts(const std::vector<int>& counts); 

    static void ExpectedSeeds(std::vector<int>& seeds, const std::vector<int>& counts );
    static int  CompareSeeds( const std::vector<int>& seeds, const std::vector<int>& xseeds ); 


    static NP* MakeCountGensteps(const std::vector<int>& photon_counts_per_genstep);
    static void GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale ); 
    static NP* GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale ) ; 

    static void SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta ); 
    static unsigned GenstepID( int ix, int iy, int iz, int iw=0 ) ; 




};




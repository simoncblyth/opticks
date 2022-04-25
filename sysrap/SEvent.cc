#include "NP.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "storch.h"
#include "ssincos.h"
#include "sc4u.h"


#include "NP.hh"
#include "PLOG.hh"
#include "SRng.hh"
#include "SStr.hh"

#include "OpticksGenstep.h"
#include "SEvent.hh"

const plog::Severity SEvent::LEVEL = PLOG::EnvLevel("SEvent", "DEBUG") ; 


NP* SEvent::MakeDemoGensteps(const char* config)
{
    NP* gs = nullptr ;
    if(     SStr::StartsWith(config, "count")) gs = MakeCountGensteps(config) ;   
    else if(SStr::StartsWith(config, "torch")) gs = MakeTorchGensteps(config) ; 
    else if(SStr::StartsWith(config, "carrier")) gs = MakeCarrierGensteps(config) ; 
    assert(gs); 

    LOG(LEVEL) 
       << " config " << ( config ? config :  "-" )
       << " gs " << ( gs ? gs->sstr() : "-" )
       ;

    return gs ; 
}



void SEvent::FillCarrierGenstep( quad6& gs )
{
    gs.q0.u = make_uint4( OpticksGenstep_PHOTON_CARRIER, 0u, 0u, 10u );   
    gs.q1.u = make_uint4( 0u,0u,0u,0u );  
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );   // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f ); // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // flag 
}
/**
SEvent::MakeCarrierGensteps
-----------------------------

Carrier gensteps are for debugging only, the genstep simply carries the 
photon with is copied into existance. 

**/

NP* SEvent::MakeCarrierGensteps(const char* config)
{
    unsigned num_gs = 10 ; 
    NP* gs = NP::Make<float>(num_gs, 6, 4 );  
    quad6* qq = (quad6*)gs->bytes() ; 
    for(unsigned i=0 ; i < num_gs ; i++ ) FillCarrierGenstep( qq[i] ) ; 
    return gs ; 
}


void SEvent::FillTorchGenstep( torch& gs, unsigned genstep_id, unsigned numphoton_per_genstep )
{
    float3 mom = make_float3( 0.f, 0.f, 1.f );  

    gs.gentype = OpticksGenstep_TORCH ; 
    gs.wavelength = 501.f ; 
    gs.mom = normalize(mom); 
    gs.radius = 50.f ; 
    gs.pos = make_float3( 0.f, 0.f, -90.f );  
    gs.time = 0.f ; 
    gs.zenith = make_float2( 0.f, 1.f );  
    gs.azimuth = make_float2( 0.f, 1.f );  
    gs.type = storchtype::Type("disc");  
    gs.mode = 255 ;    //torchmode::Type("...");  
    gs.numphoton = numphoton_per_genstep  ;   
}

NP* SEvent::MakeTorchGensteps(const char* config)
{
   // TODO: string configured gensteps, rather the the currently fixed and duplicated one  
    unsigned num_gs = 10 ; 
    unsigned numphoton_per_genstep = 100 ; 

    NP* gs = NP::Make<float>(num_gs, 6, 4 );  
    torch* tt = (torch*)gs->bytes() ; 
    for(unsigned i=0 ; i < num_gs ; i++ ) FillTorchGenstep( tt[i], i, numphoton_per_genstep ) ; 
    return gs ; 
}

/**
SEvent::MakeSeed
------------------

Normally this is done on device, here is a simple CPU implementation that.
 
**/

NP* SEvent::MakeSeed( const NP* gs )
{
    assert( gs->has_shape(-1,6,4) );  
    int num_gs = gs->shape[0] ; 
    const torch* tt = (torch*)gs->bytes() ; 

    std::vector<int> gsp(num_gs) ; 
    for(int i=0 ; i < num_gs ; i++ ) gsp[i] = tt[i].numphoton ;

    int tot_photons = 0 ; 
    for(int i=0 ; i < num_gs ; i++ ) tot_photons += gsp[i] ; 

    NP* se = NP::Make<int>( tot_photons );  
    int* sev = se->values<int>();  

    int offset = 0 ; 
    for(int i=0 ; i < num_gs ; i++) 
    {   
        int np = gsp[i] ; 
        for(int p=0 ; p < np ; p++) sev[offset+p] = i ; 
        offset += np ; 
    }   
    return se ; 
}


NP* SEvent::MakeGensteps(const std::vector<quad6>& gs ) // static 
{
    assert( gs.size() > 0); 
    NP* a = NP::Make<float>( gs.size(), 6, 4 );  
    a->read2<float>( (float*)gs.data() );  
    return a ; 
}


/**
SEvent::StandardizeCEGS
--------------------------

The cegs vector configures a grid. 
Symmetric and offset grid input configs are supported using 
vectors of length 4 and 7. 

This method standardizes the specification 
into an absolute index form which is used by 
SEvent::MakeCenterExtentGensteps

nx:ny:nz:num_photons
     symmetric grid -nx:nx, -ny:ny, -nz:nz  

nx:ny:nz:dx:dy:dz:num_photons
     offset grid -nx+dx:nx+dx, -ny+dy:ny+dy, -nz+dz:nz+dz  

ix0:iy0:iz0:ix1:iy1:iz1:num_photons 
     standardized absolute form of grid specification 


**/

void SEvent::StandardizeCEGS( const float4& ce, std::vector<int>& cegs, float gridscale ) // static 
{
    int ix0, ix1, iy0, iy1, iz0, iz1, photons_per_genstep ; 
    if( cegs.size() == 4 ) 
    {   
        cegs.resize(7) ; 

        ix0 = -cegs[0] ; ix1 = cegs[0] ; 
        iy0 = -cegs[1] ; iy1 = cegs[1] ; 
        iz0 = -cegs[2] ; iz1 = cegs[2] ; 
        photons_per_genstep = cegs[3] ;
    }   
    else if( cegs.size() == 7 ) 
    {  
        int nx = std::abs(cegs[0]) ; 
        int ny = std::abs(cegs[1]) ; 
        int nz = std::abs(cegs[2]) ; 
        int dx = cegs[3] ; 
        int dy = cegs[4] ; 
        int dz = cegs[5] ; 
        photons_per_genstep = cegs[6] ;

        ix0 = -nx + dx ; 
        iy0 = -ny + dy ; 
        iz0 = -nz + dz ; 
        ix1 =  nx + dx ; 
        iy1 =  ny + dy ; 
        iz1 =  nz + dz ; 
    }   
    else
    {   
        LOG(fatal) << " unexpected input cegs.size, expect 4 or 7 but find:" << cegs.size()  ;   
        assert(0); 
    }   

    cegs[0] = ix0 ; 
    cegs[1] = ix1 ;
    cegs[2] = iy0 ; 
    cegs[3] = iy1 ;
    cegs[4] = iz0 ;
    cegs[5] = iz1 ;
    cegs[6] = photons_per_genstep ;


    //  +---+---+---+---+
    // -2  -1   0   1   2
    
    unsigned grid_points = (ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) ;  
    unsigned tot_photons = grid_points*photons_per_genstep ; 

    LOG(info)
        << " CEGS "
        << " ix0 ix1 " << ix0 << " " << ix1
        << " iy0 iy1 " << iy0 << " " << iy1
        << " iz0 iz1 " << iz0 << " " << iz1
        << " photons_per_genstep " << photons_per_genstep
        << " grid_points (ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) " << grid_points
        << " tot_photons (grid_points*photons_per_genstep) " << tot_photons
        ;

    float3 mn ; 
    float3 mx ; 

    bool ce_offset = true ; 
    GetBoundingBox( mn, mx, ce, cegs, gridscale, ce_offset ); 
}

void SEvent::GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, bool ce_offset ) // static 
{
    assert( standardized_cegs.size() == 7 ) ; 

    int ix0 = standardized_cegs[0] ; 
    int ix1 = standardized_cegs[1] ; 
    int iy0 = standardized_cegs[2] ; 
    int iy1 = standardized_cegs[3] ; 
    int iz0 = standardized_cegs[4] ; 
    int iz1 = standardized_cegs[5] ; 
    int photons_per_genstep = standardized_cegs[6] ;


    float x0 = float(ix0)*gridscale*ce.w + ( ce_offset ? ce.x : 0.f ) ;
    float x1 = float(ix1)*gridscale*ce.w + ( ce_offset ? ce.x : 0.f ) ;

    float y0 = float(iy0)*gridscale*ce.w + ( ce_offset ? ce.y : 0.f ) ;
    float y1 = float(iy1)*gridscale*ce.w + ( ce_offset ? ce.y : 0.f ) ;

    float z0 = float(iz0)*gridscale*ce.w + ( ce_offset ? ce.z : 0.f ) ;
    float z1 = float(iz1)*gridscale*ce.w + ( ce_offset ? ce.z : 0.f ) ;

    mn.x = x0 ; 
    mx.x = x1 ; 
 
    mn.y = y0 ; 
    mx.y = y1 ; 
 
    mn.z = z0 ; 
    mx.z = z1 ; 

    LOG(LEVEL)
        << " ce_offset " << ce_offset 
        << " x0 " << std::setw(10) << std::fixed << std::setprecision(3) << x0
        << " x1 " << std::setw(10) << std::fixed << std::setprecision(3) << x1
        << " y0 " << std::setw(10) << std::fixed << std::setprecision(3) << y0
        << " y1 " << std::setw(10) << std::fixed << std::setprecision(3) << y1
        << " z0 " << std::setw(10) << std::fixed << std::setprecision(3) << z0
        << " z1 " << std::setw(10) << std::fixed << std::setprecision(3) << z1
        << " photons_per_genstep " << photons_per_genstep
        << " gridscale " << std::setw(10) << std::fixed << std::setprecision(3) << gridscale
        << " ce.w(extent) " << std::setw(10) << std::fixed << std::setprecision(3) << ce.w
        ;
}

/**
SEvent::GenstepID
-------------------

Pack four signed integers (assumed to be in char range -128 to 127) 
into a 32 bit unsigtned char using C4U uniform.  

**/

unsigned SEvent::GenstepID( int ix, int iy, int iz, int iw )
{ 
    C4U gsid ;   // sc4u.h 

    gsid.c4.x = ix ; 
    gsid.c4.y = iy ; 
    gsid.c4.z = iz ; 
    gsid.c4.w = iw ; 

    return gsid.u ; 
}



/**
SEvent::ConfigureGenstep
---------------------------

TODO: pack enums to make room for a photon_offset 

* gsid was MOVED from (1,3) to (0,2) when changing genstep to carry transform

**/

void SEvent::ConfigureGenstep( quad6& gs,  int gencode, int gridaxes, int gsid, int photons_per_genstep )
{
    assert( gencode == OpticksGenstep_TORCH ); 
    assert( gridaxes == XYZ ||  gridaxes == YZ || gridaxes == XZ || gridaxes == XY ); 

    gs.q0.i.x = gencode ;
    gs.q0.i.y = gridaxes ; 
    gs.q0.u.z = gsid ;     
    gs.q0.i.w = photons_per_genstep ;
}


/**
SEvent::MakeCenterExtentGensteps
----------------------------------
    
Creates grid of gensteps centered at ce.xyz with the grid specified 
by integer ranges that are used to scale the extent parameter to yield
offsets from the center. 
    
ce(float4)
   cx:cy:cz:extent  
    
cegs(uint4)
   nx:ny:nz:photons_per_genstep
   specifies a grid of integers -nx:nx -ny:ny -nz:nz inclusive used to scale the extent 
    
   The number of gensteps becomes: (2*nx+1)*(2*ny+1)*(2*nz+1)
    
gridscale
   float multiplier applied to the grid integers, values less than 1. (eg 0.2) 
   increase the concentration of the genstep grid on the target geometry giving a 
   better intersect rendering of a smaller region 
    
   To expand the area when using a finer grid increase the nx:ny:nz, however
   that will lead to a slower render. 


The gensteps are consumed by qsim::generate_photon_torch
Which needs to use the gensteps data in order to transform the axis 
aligned local frame grid of positions and directions 
into global frame equivalents. 

Instance transforms are best regarded as first doing rotate 
about a local origin and then translate into global position.
When wish to create multiple transforms with small local frame offsets 
to create a grid or plane between them need to first pre-multiply by the 
small local translation followed by the rotation and large global translation 
into position. 

For example when using reverse=true get individual tilts out of the plane 
ie the single local XZ plane becomes lots of planes in global frame as the local_translate is done last.
When using reverse=false get all the tilts the same so local XZ single plane stays one plane in the global 
frame as are doing the local_translate first. 

**/


NP* SEvent::MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, bool ce_offset, bool ce_scale ) // static
{
    std::vector<quad6> gensteps ;
    quad6 gs ; gs.zero();

    assert( cegs.size() == 7 );

    int ix0 = cegs[0] ;
    int ix1 = cegs[1] ;
    int iy0 = cegs[2] ;
    int iy1 = cegs[3] ;
    int iz0 = cegs[4] ;
    int iz1 = cegs[5] ;
    int photons_per_genstep = cegs[6] ;

    int nx = (ix1 - ix0)/2 ; 
    int ny = (iy1 - iy0)/2 ; 
    int nz = (iz1 - iz0)/2 ; 

    int gridaxes = GridAxes(nx, ny, nz);   // { XYZ, YZ, XZ, XY }

    LOG(info) 
        << " ce_offset " << ce_offset 
        << " ce_scale " << ce_scale
        << " nx " << nx 
        << " ny " << ny 
        << " nz " << nz 
        << " GridAxes " << gridaxes
        << " GridAxesName " << GridAxesName(gridaxes)
        ;

    /**
    where should ce offset go to handle global geom

    1. q1 of then genstep
    2. translation held in the genstep transform
    
    local frame position : currently origin, same for all gensteps : only the transform is changed   

    **/
    gs.q1.f.x = ce_offset ? ce.x : 0.f ;  
    gs.q1.f.y = ce_offset ? ce.y : 0.f ;
    gs.q1.f.z = ce_offset ? ce.z : 0.f ;
    gs.q1.f.w = 1.f ;

    double local_scale = ce_scale ? double(gridscale)*ce.w : double(gridscale) ; 
    // hmm: when using SCenterExtentFrame model2world transform the 
    // extent is already handled within the transform so must not apply extent scaling 


    unsigned photon_offset = 0 ; 

    for(int ix=ix0 ; ix < ix1+1 ; ix++ )
    for(int iy=iy0 ; iy < iy1+1 ; iy++ )
    for(int iz=iz0 ; iz < iz1+1 ; iz++ )
    {
        //LOG(LEVEL) << " ix " << ix << " iy " << iy << " iz " << iz  ;

        double tx = double(ix)*local_scale ;
        double ty = double(iy)*local_scale ;
        double tz = double(iz)*local_scale ;

        const Tran<double>* local_translate = Tran<double>::make_translate( tx, ty, tz ); 
        // grid shifts 

        bool reverse = false ;
        const Tran<double>* transform = Tran<double>::product( geotran, local_translate, reverse );

        qat4* qc = Tran<double>::ConvertFrom( transform->t ) ;

        unsigned gsid = GenstepID(ix,iy,iz,0) ; 

        ConfigureGenstep(gs, OpticksGenstep_TORCH, gridaxes, gsid, photons_per_genstep );  

        qc->write(gs);  // copy qc into gs.q2,q3,q4,q5

        gensteps.push_back(gs);
        photon_offset += std::abs(photons_per_genstep) ; 
    }
    LOG(LEVEL) << " gensteps.size " << gensteps.size() ;
    return MakeGensteps(gensteps);
}


const char* SEvent::XYZ_ = "XYZ" ; 
const char* SEvent::YZ_  = "YZ" ; 
const char* SEvent::XZ_  = "XZ" ; 
const char* SEvent::XY_  = "XY" ; 

const char* SEvent::GridAxesName( int gridaxes)  // static 
{
    const char* s = nullptr ; 
    switch( gridaxes )
    {
        case XYZ: s = XYZ_ ; break ; 
        case YZ:  s = YZ_  ; break ; 
        case XZ:  s = XZ_  ; break ; 
        case XY:  s = XY_  ; break ; 
    }
    return s ;
}

/**
SEvent::GridAxes
-----------------

The nx:ny:nz dimensions of the grid are used to classify it into::

    YZ 
    XZ  
    XY 
    XYZ

For a planar grid one of the nx:ny:nz grid dimensions is zero.
XYZ is a catch all for non-planar grids.

**/

int SEvent::GridAxes(int nx, int ny, int nz)  // static
{
    int gridaxes = XYZ ;  
    if( nx == 0 && ny > 0 && nz > 0 )
    {
        gridaxes = YZ ;  
    }
    else if( nx > 0 && ny == 0 && nz > 0 )
    {
        gridaxes = XZ ;  
    }
    else if( nx > 0 && ny > 0 && nz == 0 )
    {
        gridaxes = XY ;  
    }
    return gridaxes ; 
}



NP* SEvent::MakeCountGensteps(const char* config) // static 
{
    std::vector<int>* photon_counts_per_genstep = nullptr ; 
    if( config == nullptr )
    {
        (*photon_counts_per_genstep) = { 3, 5, 2, 0, 1, 3, 4, 2, 4 }; 
    }
    return MakeCountGensteps(*photon_counts_per_genstep);
}

/**
SEvent::MakeCountGensteps
---------------------------

Used by qudarap/tests/QEventTest.cc

**/


NP* SEvent::MakeCountGensteps(const std::vector<int>& counts) // static 
{
    int gencode = OpticksGenstep_TORCH ;
    std::vector<quad6> gensteps ;
    for(unsigned i=0 ; i < counts.size() ; i++)
    {
        quad6 gs ; gs.zero(); 

        int gridaxes = XYZ ;  
        int gsid = 0 ;  
        int photons_per_genstep = counts[i]; 

        ConfigureGenstep(gs, gencode, gridaxes, gsid, photons_per_genstep ); 

        gs.q1.f.x = 0.f ;  gs.q1.f.y = 0.f ;  gs.q1.f.z = 0.f ;  gs.q1.f.w = 0.f ;

        // identity transform to avoid nan 
        gs.q2.f.x = 1.f ;  gs.q2.f.y = 0.f ;  gs.q2.f.z = 0.f ;  gs.q2.f.w = 0.f ;
        gs.q3.f.x = 0.f ;  gs.q3.f.y = 1.f ;  gs.q3.f.z = 0.f ;  gs.q3.f.w = 0.f ;
        gs.q4.f.x = 0.f ;  gs.q4.f.y = 0.f ;  gs.q4.f.z = 1.f ;  gs.q4.f.w = 0.f ;
        gs.q5.f.x = 0.f ;  gs.q5.f.y = 0.f ;  gs.q5.f.z = 0.f ;  gs.q5.f.w = 1.f ;

        gensteps.push_back(gs);
    }
    return MakeGensteps(gensteps);
}

unsigned SEvent::SumCounts(const std::vector<int>& counts) // static 
{
    unsigned total = 0 ; 
    for(unsigned i=0 ; i < counts.size() ; i++) total += counts[i] ; 
    return total ; 
}


/**
SEvent::ExpectedSeeds
----------------------

From a vector of counts populate the vector of seeds by simple CPU side duplication.  

**/

void SEvent::ExpectedSeeds(std::vector<int>& seeds, const std::vector<int>& counts ) // static 
{
    unsigned total = SumCounts(counts);  
    unsigned ni = counts.size(); 
    for(unsigned i=0 ; i < ni ; i++)
    {   
        int np = counts[i] ; 
        for(int p=0 ; p < np ; p++) seeds.push_back(i) ; 
    }   
    assert( seeds.size() == total );  
}

int SEvent::CompareSeeds( const std::vector<int>& seeds, const std::vector<int>& xseeds ) // static 
{
    assert( seeds.size() == xseeds.size() );  
    int mismatch = 0 ; 
    for(unsigned i=0 ; i < seeds.size() ; i++) if( seeds[i] != xseeds[i] ) mismatch += 1 ; 
    return mismatch ; 
}







/**
SEvent::GenerateCenterExtentGenstepsPhotons
---------------------------------------------

**/

NP* SEvent::GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale )
{
    std::vector<quad4> pp ;
    SEvent::GenerateCenterExtentGenstepsPhotons( pp, gsa, gridscale ); 

    NP* ppa = NP::Make<float>( pp.size(), 4, 4 );
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );
    return ppa ; 
}


/**
SEvent::GenerateCenterExtentGenstepsPhotons
---------------------------------------------

Contrast this CPU implementation of CEGS generation with qudarap/qsim.h qsim<T>::generate_photon_torch

**/

void SEvent::GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale )
{
    LOG(info) << " gsa " << gsa->sstr() ; 
    assert( gsa->shape.size() == 3 && gsa->shape[1] == 6 && gsa->shape[2] == 4 );

    std::vector<quad6> gsv(gsa->shape[0]) ; 
    memcpy( gsv.data(), gsa->bytes(), gsa->arr_bytes() );
 
    quad4 p ;
    p.zero();
    
    unsigned seed = 0 ; 
    SRng<float> rng(seed) ;


    float3 paradir ; 
    qvals(paradir, "PARADIR", "0,0,0" );  
    bool with_paradir = dot(paradir,paradir) > 0.f ; 
    if(with_paradir)  
    {
        paradir = normalize(paradir);     
        LOG(info) << " PARADIR enabled " << paradir ; 
    }
    else
    {
        LOG(info) << " PARADIR NOT-enabled " ; 
    }

    
    for(unsigned i=0 ; i < gsv.size() ; i++)
    {   
        const quad6& gs = gsv[i]; 
        qat4 qt(gs) ;  // copy 4x4 transform from last 4 quads of genstep 

        C4U gsid ;   // genstep integer grid coordinate IXYZ and IW photon index up to 255

        int gencode           = gs.q0.i.x ; 
        int gridaxes          = gs.q0.i.y ; 
        gsid.u                = gs.q0.u.z ;     // formerly gs.q1.u.w 
        int      num_photons_ = gs.q0.i.w ; 
        unsigned num_photons  = std::abs(num_photons_);  

        assert( gencode == OpticksGenstep_TORCH );

        //std::cout << " i " << i << " num_photons " << num_photons << std::endl ;
        
        double u0, u1 ; 
        double phi, sinPhi,   cosPhi ; 
        double sinTheta, cosTheta ; 
        
        for(unsigned j=0 ; j < num_photons ; j++)
        {   
            // this inner loop should be similar to quadarap/qsim.h/generate_photon_torch
            // TODO: arrange a header such that can actually use the same code via some curand_uniform macro refinition trickery
            // -ve num_photons uses regular, not random azimuthal spray of directions

            u0 = num_photons_ < 0 ? double(j)/double(num_photons-1) : rng() ;

            phi = 2.*M_PIf*u0 ;     // azimuthal 0->2pi 
            ssincos(phi,sinPhi,cosPhi);  

            
            // cosTheta sinTheta are only used for 3D (not 2D planar gensteps)
            u1 = rng(); 
            cosTheta = u1 ; 
            sinTheta = sqrtf(1.0-u1*u1);

            if( with_paradir )
            {
                // what scaling is needed to span the grid ?
                p.q0.f.x = 0.f ;  
                p.q0.f.y = u0*float(num_photons-1)*gridscale ; 
                p.q0.f.z = 0.f ; 
                p.q0.f.w = 1.f ;  

                p.q1.f.x = paradir.x ; 
                p.q1.f.y = paradir.y ; 
                p.q1.f.z = paradir.z ; 
                p.q1.f.w = 0.f ; 
            }
            else
            {
                // copy position from genstep into the photon, historically has been origin   
                p.q0.f.x = gs.q1.f.x ;
                p.q0.f.y = gs.q1.f.y ;
                p.q0.f.z = gs.q1.f.z ;
                p.q0.f.w = 1.f ;       // <-- dont copy the "float" gsid 

                SetGridPlaneDirection( p.q1.f, gridaxes, cosPhi, sinPhi, cosTheta, sinTheta );  
            }

            // tranform photon position and direction into the desired frame
            qt.right_multiply_inplace( p.q0.f, 1.f );  // position 
            qt.right_multiply_inplace( p.q1.f, 0.f );  // direction

            unsigned char ucj = (j < 255 ? j : 255 ) ;  // photon index local to the genstep
            gsid.c4.w = ucj ;     // setting C4U union element to change gsid.u 
            p.q3.u.w = gsid.u ;   // include photon index IW with the genstep coordinate into photon (3,3)
        
            pp.push_back(p) ;
        }
    }

    LOG(info) << " pp.size " << pp.size() ; 
}


/**
SEvent::SetGridPlaneDirection
----------------------------------

The cosTheta and sinTheta arguments are only used for the 3D gridaxes == XYZ

**/

void SEvent::SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta )
{
    if( gridaxes == YZ )
    {    
        dir.x = 0.f ;  
        dir.y = float(cosPhi) ;
        dir.z = float(sinPhi) ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XZ )
    {    
        dir.x = float(cosPhi) ;   
        dir.y = 0.f    ;
        dir.z = float(sinPhi) ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XY )
    {    
        dir.x = float(cosPhi) ;   
        dir.y = float(sinPhi) ;
        dir.z = 0.f     ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XYZ )    // formerly adhoc used XZ here
    {    
        dir.x = float(sinTheta * cosPhi)  ; 
        dir.y = float(sinTheta * sinPhi)  ; 
        dir.z = float(cosTheta) ; 
        dir.w =  0.f   ; 
    } 
    else
    {
        LOG(fatal) << " invalid gridaxes value " << gridaxes ; 
        assert(0);  
    }
}


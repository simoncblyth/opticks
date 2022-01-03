#include "NP.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "ssincos.h"

#include "NP.hh"
#include "PLOG.hh"
#include "SRng.hh"

#include "OpticksGenstep.h"
#include "SEvent.hh"

const plog::Severity SEvent::LEVEL = PLOG::EnvLevel("SEvent", "DEBUG") ; 

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
        int dx = cegs[3] ; 
        int dy = cegs[4] ; 
        int dz = cegs[5] ; 
        photons_per_genstep = cegs[6] ;

        ix0 = -cegs[0] + dx ; 
        iy0 = -cegs[1] + dy ; 
        iz0 = -cegs[2] + dz ; 
        ix1 =  cegs[0] + dx ; 
        iy1 =  cegs[1] + dy ; 
        iz1 =  cegs[2] + dz ; 
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

    LOG(info)
        << " CXS_CEGS "
        << " ix0 ix1 " << ix0 << " " << ix1
        << " iy0 iy1 " << iy0 << " " << iy1
        << " iz0 iz1 " << iz0 << " " << iz1
        << " photons_per_genstep " << photons_per_genstep
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

    int gridaxes = GridAxes(nx, ny, nz); 
    int dirmode = gridaxes == XYZ ? DIMENSION_3 : DIMENSION_2 ; 

    LOG(LEVEL) 
        << " ce_offset " << ce_offset 
        << " ce_scale " << ce_scale
        << " nx " << nx 
        << " ny " << ny 
        << " nz " << nz 
        << " GridAxes " << gridaxes
        << " GridAxesName " << GridAxesName(gridaxes)
        << " DirMode " << dirmode
        << " DirModeName " << DirModeName(dirmode)
        ;

    gs.q0.i.x = OpticksGenstep_TORCH ;
    gs.q0.i.y = gridaxes ; 
    gs.q0.i.z = dirmode ; 
    gs.q0.i.w = photons_per_genstep ;

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

    double local_scale = ce_scale ?double(gridscale)*ce.w : double(gridscale) ; 
    // hmm: when using SCenterExtentFrame model2world transform the 
    // extent is already handled within the transform so must not apply extent scaling 


    for(int ix=ix0 ; ix < ix1+1 ; ix++ )
    for(int iy=iy0 ; iy < iy1+1 ; iy++ )
    for(int iz=iz0 ; iz < iz1+1 ; iz++ )
    {
        LOG(LEVEL) << " ix " << ix << " iy " << iy << " iz " << iz  ;

        double tx = double(ix)*local_scale ;
        double ty = double(iy)*local_scale ;
        double tz = double(iz)*local_scale ;

        const Tran<double>* local_translate = Tran<double>::make_translate( tx, ty, tz ); 
        // grid shifts 
 

        bool reverse = false ;
        const Tran<double>* transform = Tran<double>::product( geotran, local_translate, reverse );

        qat4* qc = Tran<double>::ConvertFrom( transform->t ) ;

        qc->write(gs);                    // copy qc into gs.q2,q3,q4,q5

        gensteps.push_back(gs);
    }
    LOG(LEVEL) << " gensteps.size " << gensteps.size() ;

    return MakeGensteps(gensteps);
}



const char* SEvent::DIMENSION_3_ = "3D" ; 
const char* SEvent::DIMENSION_2_ = "2D" ; 
const char* SEvent::DIMENSION_1_ = "1D" ; 
const char* SEvent::DIMENSION_0_ = "0D" ;
 
const char* SEvent::DirModeName( int dirmode )  // static 
{
    const char* s = nullptr ; 
    switch( dirmode )
    {
        case DIMENSION_3: s = DIMENSION_3_ ; break ; 
        case DIMENSION_2: s = DIMENSION_2_ ; break ; 
        case DIMENSION_1: s = DIMENSION_1_ ; break ; 
        case DIMENSION_0: s = DIMENSION_0_ ; break ; 
    }
    return s ;
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



NP* SEvent::MakeCountGensteps() // static 
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    return MakeCountGensteps(photon_counts_per_genstep);
}
NP* SEvent::MakeCountGensteps(const std::vector<int>& counts) // static 
{
    std::vector<quad6> gs ;
    for(unsigned i=0 ; i < counts.size() ; i++)
    {
        int gencode = OpticksGenstep_TORCH ;
        quad6 qq ;
        qq.q0.i.x = gencode  ;   qq.q0.i.y = -1 ;   qq.q0.i.z = -1 ;   qq.q0.i.w = counts[i] ;
        qq.q1.f.x = 0.f ;  qq.q1.f.y = 0.f ;  qq.q1.f.z = 0.f ;   qq.q1.f.w = 0.f ;
        qq.q2.i.x = -1 ;   qq.q2.i.y = -1 ;   qq.q2.i.z = -1 ;   qq.q2.i.w = -1 ;
        qq.q3.i.x = -1 ;   qq.q3.i.y = -1 ;   qq.q3.i.z = -1 ;   qq.q3.i.w = -1 ;
        qq.q4.i.x = -1 ;   qq.q4.i.y = -1 ;   qq.q4.i.z = -1 ;   qq.q4.i.w = -1 ;
        qq.q5.i.x = -1 ;   qq.q5.i.y = -1 ;   qq.q5.i.z = -1 ;   qq.q5.i.w = -1 ;
        gs.push_back(qq);
    }
    return MakeGensteps(gs);
}

/**
SEvent::GenerateCenterExtentGenstepsPhotons
---------------------------------------------

**/

NP* SEvent::GenerateCenterExtentGenstepsPhotons_( const NP* gsa )
{
    std::vector<quad4> pp ;
    SEvent::GenerateCenterExtentGenstepsPhotons( pp, gsa ); 

    NP* ppa = NP::Make<float>( pp.size(), 4, 4 );
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );
    return ppa ; 
}


/**
SEvent::GenerateCenterExtentGenstepsPhotons
---------------------------------------------

Contrast this CPU implementation of CEGS generation with qudarap/qsim.h qsim<T>::generate_photon_torch

**/

void SEvent::GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa )
{
    LOG(info) << " gsa " << gsa->sstr() ; 
    assert( gsa->shape.size() == 3 && gsa->shape[1] == 6 && gsa->shape[2] == 4 );

    std::vector<quad6> gsv(gsa->shape[0]) ; 
    memcpy( gsv.data(), gsa->bytes(), gsa->arr_bytes() );
 
    quad4 p ;
    p.zero();
    
    unsigned seed = 0 ; 
    SRng<float> rng(seed) ;
    
    for(unsigned i=0 ; i < gsv.size() ; i++)
    {   
        const quad6& gs = gsv[i]; 
        qat4 qt(gs) ;  // copy 4x4 transform from last 4 quads of genstep 
        
        unsigned num_photons = gs.q0.u.w ; 
        int gridaxes = gs.q0.i.y ; 
        int dirmode  = gs.q0.i.z ; 

        //std::cout << " i " << i << " num_photons " << num_photons << std::endl ;
        
        double u0, phi  , sinPhi,   cosPhi ; 
        double u1, sinTheta, cosTheta ; 
        
        for(unsigned j=0 ; j < num_photons ; j++)
        {   
            u0 = rng();
            //u0 = double(j)/double(num_photons-1) ;

            u1 = rng(); 

            phi = 2.*M_PIf*u0 ;     // azimuthal 0->2pi 
            ssincos(phi,sinPhi,cosPhi);  

            cosTheta = u1 ; 
            sinTheta = sqrtf(1.0-u1*u1);

            p.q0.f = gs.q1.f ;  // copy position from genstep into the photon, historically has been origin   
 
            switch(dirmode)
            {
                case DIMENSION_2: SetGridPlaneDirection_2D( p.q1.f, gridaxes, cosPhi, sinPhi )                    ; break ; 
                case DIMENSION_3: SetGridPlaneDirection_3D( p.q1.f, gridaxes, cosPhi, sinPhi, cosTheta, sinTheta ); break ; 
            }

            // tranforming photon position and direction into the desired frame
 
            qt.right_multiply_inplace( p.q0.f, 1.f );  // position 
            qt.right_multiply_inplace( p.q1.f, 0.f );  // direction
            
            pp.push_back(p) ;
        }
    }

    LOG(info) << " pp.size " << pp.size() ; 
}


/**
SEvent::SetGridPlaneDirection_2D
----------------------------------

TODO: probably need some minus signs in the below for consistency 
TODO: for gridaxes XYZ an adhoc choice is made for the plane of the direction 

**/

void SEvent::SetGridPlaneDirection_2D( float4& dir, int gridaxes, float cosPhi, float sinPhi )
{
    if( gridaxes == YZ )
    {    
        dir.x = 0.f ;  
        dir.y = cosPhi ;
        dir.z = sinPhi ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XZ )
    {    
        dir.x = cosPhi ;   
        dir.y = 0.f    ;
        dir.z = sinPhi ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XY )
    {    
        dir.x = cosPhi ;   
        dir.y = sinPhi ;
        dir.z = 0.     ;   
        dir.w = 0.f    ;   
    }
    else if( gridaxes == XYZ )
    {    
        dir.x = cosPhi ;   
        dir.y = 0.f    ;
        dir.z = sinPhi ;   
        dir.w = 0.f    ;   
    } 
    else
    {
        LOG(fatal) << " invalid gridaxes value " << gridaxes ; 
        assert(0);  
    }
}


void SEvent::SetGridPlaneDirection_3D( float4& dir, int gridaxes, float cosPhi, float sinPhi, float cosTheta, float sinTheta ) // static 
{
    assert( gridaxes == XYZ ); 
    dir.x =  sinTheta * cosPhi  ; 
    dir.y =  sinTheta * sinPhi  ; 
    dir.z =  cosTheta ; 
    dir.w =  0.   ; 
}





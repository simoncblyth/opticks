#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "sframe.h"

#include "sc4u.h"
#include "ssincos.h"

#include "SLOG.hh"
#include "SSys.hh"
#include "SRng.hh"
#include "SGenstep.hh"
#include "OpticksGenstep.h"
#include "SFrameGenstep.hh"
#include "NP.hh"


const plog::Severity SFrameGenstep::LEVEL = SLOG::EnvLevel("SFrameGenstep", "DEBUG" ); 

/**
SFrameGenstep::CE_OFFSET
-------------------------

Interprets the CE_OFFSET envvar converting the values from the string (default "0,0,0")
into the ce_offset vector with one or more float3.

Typically CE_OFFSET "0.,0.,0." corresponding to local frame origin.
The string "CE" or "ce" in the envvar is special cased, causing 
the offset to be set to the ce provided by the argument.  

**/

void SFrameGenstep::CE_OFFSET(std::vector<float3>& ce_offset, const float4& ce ) // static
{
    const char* ekey = "CE_OFFSET" ; 
    const char* val = SSys::getenvvar(ekey); 

    bool is_CE = val == nullptr ? false : ( strcmp(val, "CE")== 0 || strcmp(val, "ce")== 0 )  ; 
    float3 offset = make_float3(0.f, 0.f, 0.f ); 

    if(is_CE)   // this is not typically used anymore
    {
        offset.x = ce.x ; 
        offset.y = ce.y ; 
        offset.z = ce.z ; 
        ce_offset.push_back(offset); 
    }
    else
    {
        std::vector<float>* fvec = SSys::getenvfloatvec(ekey, "0,0,0"); 
        unsigned num_values = fvec->size() ;  
        assert(fvec); 

        bool multiple_of_3 = num_values % 3 == 0 ;
        LOG_IF(fatal, !multiple_of_3) << " not multiple_of_3 num_values " << num_values << " ekey " << ekey  ; 
        assert(multiple_of_3); 
        unsigned num_offset = num_values/3 ; 
        for(unsigned i=0 ; i < num_offset ; i++)
        {
            offset.x = (*fvec)[i*3+0] ; 
            offset.y = (*fvec)[i*3+1] ; 
            offset.z = (*fvec)[i*3+2] ; 
            ce_offset.push_back(offset); 
        }
    }

    LOG(LEVEL) 
         << "ekey " << ekey 
         << " val " << val 
         << " is_CE " << is_CE
         << " ce_offset.size " << ce_offset.size() 
         << " ce " << ce 
         ; 

    LOG(LEVEL) << Desc(ce_offset) ; 

}

std::string SFrameGenstep::Desc(const std::vector<float3>& ce_offset )
{
    std::stringstream ss ; 
    ss << "SFrameGenstep::Desc ce_offset.size " << ce_offset.size() << std::endl ; 
    for(unsigned i=0 ; i < ce_offset.size() ; i++) 
    {
        const float3& offset = ce_offset[i] ; 
        ss << std::setw(4) << i << " : " << offset << std::endl ;   
    }

    std::string s = ss.str(); 
    return s ; 
}

std::string SFrameGenstep::Desc(const std::vector<int>& cegs )
{
    std::stringstream ss ; 
    ss << " size " << cegs.size() << "[" ; 
    for(unsigned i=0 ; i < cegs.size() ; i++) ss << cegs[i] << " " ; 
    ss << "]" ; 
    std::string s = ss.str(); 
    return s ; 
}



void SFrameGenstep::GetGridConfig(std::vector<int>& cegs,  const char* ekey, char delim, const char* fallback )
{
    SSys::getenvintvec(ekey, cegs, delim, fallback );
    StandardizeCEGS(cegs); 
    assert( cegs.size() == 0 || cegs.size() == 8 );
    LOG(LEVEL) << " ekey " << ekey << " Desc " << Desc(cegs)  ; 
}



/**
SFrameGenstep::MakeCenterExtentGensteps
-----------------------------------------

Canonical usage is from SEvt::setFrame for simtrace running. 

NOTE: changed CE_SCALE default to be 1, enabling it
To switch off CE scaling set CE_SCALE envvar to "0" 

HMM as the typical CEGS is 16:0:9:500 to conform to standard aspect 
ratio GRIDSCALE of 0.1 is more resonable than 1 ? 

The sframe argument is used for its *ce* and the details
of the grid config are saved into the sframe 
which makes the config available from python. 

**/


NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
{
    const float4& ce = fr.ce ; 
    float gridscale = SSys::getenvfloat("GRIDSCALE", 0.1 ) ; 

    // CSGGenstep::init
    std::vector<int> cegs ; 
    GetGridConfig(cegs, "CEGS", ':', "16:0:9:1000" ); 
    fr.set_grid(cegs, gridscale); 

    std::vector<float3> ce_offset ; 
    CE_OFFSET(ce_offset, ce); 

    LOG(LEVEL) 
        << " ce " << ce 
        << " ce_offset.size " << ce_offset.size() 
        ;


    int ce_scale = SSys::getenvint("CE_SCALE", 1) ; // TODO: ELIMINATE AFTER RTP CHECK 
    LOG_IF(fatal, ce_scale == 0) << "warning CE_SCALE is not enabled : NOW THINK THIS SHOULD ALWAYS BE ENABLED " ; 
 

    //Tran<double>* geotran = Tran<double>::FromPair( &fr.m2w, &fr.w2m, 1e-6 ); 
    Tran<double>* geotran = fr.getTransform(); 


    std::vector<NP*> gsl ; 
    NP* gs_base = MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
    gsl.push_back(gs_base) ; 

    std::vector<std::string> keys = {{
        "CEHIGH_0", 
        "CEHIGH_1", 
        "CEHIGH_2", 
        "CEHIGH_3",
        "CEHIGH_4",
        "CEHIGH_5",
        "CEHIGH_6",
        "CEHIGH_7",
        "CEHIGH_8",
        "CEHIGH_9"
      }} ; 

    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const char* key = keys[i].c_str() ; 
        std::vector<int> cehigh ; 
        GetGridConfig(cehigh, key, ':', "" ); 
        LOG(LEVEL) << " key " << key << " cehigh.size " << cehigh.size() ;  
        if(cehigh.size() == 8)
        {
            NP* gs_cehigh = MakeCenterExtentGensteps(ce, cehigh, gridscale, geotran, ce_offset, ce_scale );
            gsl.push_back(gs_cehigh) ; 
        }
    }

    LOG(LEVEL) << " gsl.size " << gsl.size() ; 

    LOG(LEVEL) << "[ NP::Concatenate " ; 
    NP* gs = NP::Concatenate(gsl) ; 
    LOG(LEVEL) << "] NP::Concatenate " ; 

    gs->set_meta<int>("midx", fr.midx() );
    gs->set_meta<int>("mord", fr.mord() );
    gs->set_meta<int>("iidx", fr.iidx() );
    gs->set_meta<float>("gridscale", fr.gridscale() );
    gs->set_meta<int>("ce_scale", int(ce_scale) ); 

    return gs ; 
}

/**
SFrameGenstep::MakeCenterExtentGensteps
-----------------------------------------
    
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

ce_offset:vector of float3 
   typically local frame origin (0.,0.,0.) 
   the grid is repeated for each offset float3 
   This is used for example to create intesects in multiple planes.  

ce_scale:true
   grid translate offsets *local_scale* set to ce.w*gridscale 

   SEEMS LIKE THIS SHOULD ALWAYS BE USED ?
   ACTUALLTY NOT : APPARENTLY SOME TYPES OF TRANSFORM INCORPORATE THE EXTENT
   SCALE ALREADY 


ce_scale:false
   grid translate offsets *local_scale* set to gridscale 



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

The *high = cegs[7]* parameter is normally 1.  For high resolution regions it is 2/3/4 
which multiplies the grid points and divides the gridscale. 
This allows high resolution regions without changes to indexing. 

**/

NP* SFrameGenstep::MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, const std::vector<float3>& ce_offset, bool ce_scale ) // static
{
    std::vector<quad6> gensteps ;
    quad6 gs ; gs.zero();

    assert( cegs.size() == 8 );

    int high = cegs[7] ; 
    assert( high == 1 || high == 2 || high == 3 || high == 4 ); 
    double scale = double(gridscale)/double(high) ; 

    int ix0 = cegs[0]*high ;
    int ix1 = cegs[1]*high ;
    int iy0 = cegs[2]*high ;
    int iy1 = cegs[3]*high ;
    int iz0 = cegs[4]*high ;
    int iz1 = cegs[5]*high ;

    int photons_per_genstep = cegs[6] ;


    int nx = (ix1 - ix0)/2 ; 
    int ny = (iy1 - iy0)/2 ; 
    int nz = (iz1 - iz0)/2 ; 

    int gridaxes = SGenstep::GridAxes(nx, ny, nz);   // { XYZ, YZ, XZ, XY }
    int num_offset = int(ce_offset.size()) ; 

    LOG(LEVEL) 
        << " num_offset " << num_offset
        << " ce_scale " << ce_scale
        << " nx " << nx 
        << " ny " << ny 
        << " nz " << nz 
        << " GridAxes " << gridaxes
        << " GridAxesName " << SGenstep::GridAxesName(gridaxes)
        << " high " << high
        << " gridscale " << gridscale
        << " scale " << scale 
        ;

    double local_scale = ce_scale ? scale*ce.w : scale ; // ce_scale:true is almost always expected 
    // hmm: when using SCenterExtentFrame model2world transform the 
    // extent is already handled within the transform so must not apply extent scaling 
    // THIS IS CONFUSING : TODO FIND WAY TO AVOID THE CONFUSION BY MAKING THE DIFFERENT TYPES OF TRANSFORM MORE CONSISTENT

    unsigned photon_offset = 0 ; 


    for(int ip=0 ; ip < num_offset ; ip++)   // planes
    {
        const float3& offset = ce_offset[ip] ; 

        gs.q1.f.x = offset.x ; 
        gs.q1.f.y = offset.y ; 
        gs.q1.f.z = offset.z ; 
        gs.q1.f.w = 1.f ;

        for(int ix=ix0 ; ix < ix1+1 ; ix++ )
        for(int iy=iy0 ; iy < iy1+1 ; iy++ )
        for(int iz=iz0 ; iz < iz1+1 ; iz++ )
        {
            double tx = double(ix)*local_scale ;
            double ty = double(iy)*local_scale ;
            double tz = double(iz)*local_scale ;

            const Tran<double>* local_translate = Tran<double>::make_translate( tx, ty, tz ); 
            // grid shifts 

            bool reverse = false ;
            const Tran<double>* transform = Tran<double>::product( geotran, local_translate, reverse );

            qat4* qc = Tran<double>::ConvertFrom( transform->t ) ;

            unsigned gsid = SGenstep::GenstepID(ix,iy,iz,ip) ; 

            SGenstep::ConfigureGenstep(gs, OpticksGenstep_FRAME, gridaxes, gsid, photons_per_genstep );  

            qc->write(gs);  // copy qc into gs.q2,q3,q4,q5

            gensteps.push_back(gs);
            photon_offset += std::abs(photons_per_genstep) ; 
        }
    }

    LOG(LEVEL) 
         << " num_offset " << num_offset 
         << " gensteps.size " << gensteps.size() 
         ;
    return SGenstep::MakeArray(gensteps);
}



/**
SFrameGenstep::StandardizeCEGS
--------------------------------

The cegs vector configures a grid. 
Symmetric and offset grid input configs are supported using 
vectors of length 4 and 7. 

This method standardizes the specification 
into an absolute index form which is used by 
SFrameGenstep::MakeCenterExtentGensteps

nx:ny:nz:num_photons
     symmetric grid -nx:nx, -ny:ny, -nz:nz  

nx:ny:nz:dx:dy:dz:num_photons
     offset grid -nx+dx:nx+dx, -ny+dy:ny+dy, -nz+dz:nz+dz  

ix0:ix1:iy0:iy1:iz0:iz1:num_photons:high
     standardized absolute form of grid specification 
     that SFrameGenstep::StandardizeCEGS creates.

     This eight element form is also now used as an
     input form when wishing to increase resolution with 
     the "high" 8th element cegs[7]. 

Examples of CEHIGH envvars using the third standardized form::

    export CEHIGH_0=-11:-9:0:0:-2:0:1000:4
    export CEHIGH_1=9:11:0:0:-2:0:1000:4
    export CEHIGH_2=-1:1:0:0:-2:0:1000:4


**/

void SFrameGenstep::StandardizeCEGS( std::vector<int>& cegs ) // static 
{
    if( cegs.size() == 0 ) return ; 
    if( cegs.size() == 8 ) return ;  // when start with 8, just exit

    bool expect = cegs.size() == 4 || cegs.size() == 7 ; 
    LOG_IF(error, !expect) << " unexpected cegs.size " << cegs.size() ; 
    assert( expect ); 

    int ix0 = 0 ; 
    int ix1 = 0 ; 
    int iy0 = 0 ; 
    int iy1 = 0 ; 
    int iz0 = 0 ; 
    int iz1 = 0 ; 
    int photons_per_genstep = 0 ;

    if( cegs.size() == 4 ) 
    {   
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

    cegs.resize(8) ; 
    cegs[0] = ix0 ; 
    cegs[1] = ix1 ;
    cegs[2] = iy0 ; 
    cegs[3] = iy1 ;
    cegs[4] = iz0 ;
    cegs[5] = iz1 ;
    cegs[6] = photons_per_genstep ;
    cegs[7] = 1 ;   // high 

    //  +---+---+---+---+
    // -2  -1   0   1   2
    
    unsigned grid_points = (ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) ;  
    unsigned tot_photons = grid_points*std::abs(photons_per_genstep) ; 

    LOG(LEVEL)
        << " CEGS "
        << " ix0 ix1 " << ix0 << " " << ix1
        << " iy0 iy1 " << iy0 << " " << iy1
        << " iz0 iz1 " << iz0 << " " << iz1
        << " photons_per_genstep " << photons_per_genstep
        << " grid_points (ix1-ix0+1)*(iy1-iy0+1)*(iz1-iz0+1) " << grid_points
        << " tot_photons (grid_points*photons_per_genstep) " << tot_photons
        ;
}


/**
SFrameGenstep::GetBoundingBox
-------------------------------

Uses CE center-extent and gridscale together with the cegs grid parameters to provide 
the float3 mn and mx bounds of the CE grid.  NB no use of any transforms here.   

**/

void SFrameGenstep::GetBoundingBox( float3& mn, float3& mx, const float4& ce, const std::vector<int>& standardized_cegs, float gridscale, const float3& ce_offset ) // static 
{
    assert( standardized_cegs.size() == 7 ) ; 

    int ix0 = standardized_cegs[0] ; 
    int ix1 = standardized_cegs[1] ; 
    int iy0 = standardized_cegs[2] ; 
    int iy1 = standardized_cegs[3] ; 
    int iz0 = standardized_cegs[4] ; 
    int iz1 = standardized_cegs[5] ; 
    int photons_per_genstep = standardized_cegs[6] ;


    float x0 = float(ix0)*gridscale*ce.w + ce_offset.x ;   // ce_offset is usually local fram origin (0.f,0.f,0.f)
    float x1 = float(ix1)*gridscale*ce.w + ce_offset.x ;

    float y0 = float(iy0)*gridscale*ce.w + ce_offset.y ; 
    float y1 = float(iy1)*gridscale*ce.w + ce_offset.y ;

    float z0 = float(iz0)*gridscale*ce.w + ce_offset.z ;
    float z1 = float(iz1)*gridscale*ce.w + ce_offset.z ;

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
SFrameGenstep::GenerateCenterExtentGenstepsPhotons
-----------------------------------------------------

**/

NP* SFrameGenstep::GenerateCenterExtentGenstepsPhotons_( const NP* gsa, float gridscale )
{
    std::vector<quad4> pp ;
    GenerateCenterExtentGenstepsPhotons( pp, gsa, gridscale ); 

    NP* ppa = NP::Make<float>( pp.size(), 4, 4 );
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );
    return ppa ; 
}




/**
SFrameGenstep::GenerateCenterExtentGenstepsPhotons
---------------------------------------------

Contrast this CPU implementation of CEGS generation with qudarap/qsim.h qsim<T>::generate_photon_torch

The inner loop should be similar to qudarap/qsim.h/generate_photon_simtrace

TODO: arrange a header such that can actually use the same code via some curand_uniform 
macro refinition trickery

-ve num_photons uses regular, not random azimuthal spray of directions

TODO: consolidate the below two methods, paradir is the hangup 


**/

void SFrameGenstep::GenerateCenterExtentGenstepsPhotons( std::vector<quad4>& pp, const NP* gsa, float gridscale )
{
    LOG(LEVEL) << " gsa " << gsa->sstr() ; 

    assert( gsa->shape.size() == 3 && gsa->shape[1] == 6 && gsa->shape[2] == 4 );
    assert( gsa->has_shape(-1,6,4) ); 

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
        LOG(LEVEL) << " PARADIR enabled " << paradir ; 
    }
    else
    {
        LOG(LEVEL) << " PARADIR NOT-enabled " ; 
    }

    
    for(unsigned i=0 ; i < gsv.size() ; i++)
    {   
        const quad6& gs = gsv[i]; 
        qat4 qt(gs) ;  // copy 4x4 transform from last 4 quads of genstep 

        C4U gsid ;   // genstep integer grid coordinate IXYZ and IW photon index up to 255

        int gencode           = gs.q0.i.x ; 
        int gridaxes          = gs.q0.i.y ; 
        gsid.u                = gs.q0.u.z ;     // formerly gs.q1.u.w 
        int      num_photons_ = gs.q0.i.w ;     // -ve num_photons_ doesnt use random u0 so creates phi wheels

        unsigned num_photons  = std::abs(num_photons_);  

        bool expect = gencode == OpticksGenstep_TORCH || gencode == OpticksGenstep_FRAME ; 

        LOG_IF(error, !expect) 
            << "unexpected gencod " << gencode 
            << " OpticksGenstep_::Name " << OpticksGenstep_::Name(gencode) ; 

        assert(expect);

        //std::cout << " i " << i << " num_photons " << num_photons << std::endl ;
        
        double u0, u1 ; 
        double phi, sinPhi,   cosPhi ; 
        double sinTheta, cosTheta ; 

        
        
        for(unsigned j=0 ; j < num_photons ; j++)
        {   
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

    LOG(LEVEL) << " pp.size " << pp.size() ; 
}


/**
SFrameGenstep::GenerateSimtracePhotons
-----------------------------------------

Canonical invokation is from SEvt::setFrame_HostsideSimtrace
using the genstep created by SFrameGenstep::MakeCenterExtentGensteps

**/

void SFrameGenstep::GenerateSimtracePhotons( std::vector<quad4>& simtrace, const std::vector<quad6>& genstep )
{
    LOG(LEVEL) 
        << " genstep.size " << genstep.size() 
        << " simtrace.size " << simtrace.size() 
        ; 

    unsigned seed = 0 ; 
    SRng<float> rng(seed) ;

    bool simtrace_layout = true ; 
    unsigned count = 0 ; 

    for(unsigned i=0 ; i < genstep.size() ; i++)
    {   
        const quad6& gs = genstep[i]; 
        C4U gsid ;   // genstep integer grid coordinate IXYZ and IW photon index up to 255
        int gencode           = gs.q0.i.x ; 
        int gridaxes          = gs.q0.i.y ; 
        gsid.u                = gs.q0.u.z ;     // formerly gs.q1.u.w 
        int      num_photons_ = gs.q0.i.w ;     // -ve num_photons_ doesnt use random u0 so creates phi wheels

        qat4 qt(gs) ;  // copy 4x4 transform from last 4 quads of genstep 

        unsigned num_photons  = std::abs(num_photons_);  
        bool expect = gencode == OpticksGenstep_TORCH || gencode == OpticksGenstep_FRAME ; 

        LOG_IF(error, !expect) 
            << "unexpected gencode " << gencode 
            << " OpticksGenstep_::Name " << OpticksGenstep_::Name(gencode) ; 

        assert(expect);
        //std::cout << " i " << i << " num_photons " << num_photons << std::endl ;
        
        double u0, u1 ; 
        double phi, sinPhi,   cosPhi ; 
        double sinTheta, cosTheta ; 

        for(unsigned j=0 ; j < num_photons ; j++)
        {   
            u0 = num_photons_ < 0 ? double(j)/double(num_photons-1) : rng() ;

            phi = 2.*M_PIf*u0 ;     // azimuthal 0->2pi 
            ssincos(phi,sinPhi,cosPhi);  
            
            // cosTheta sinTheta are only used for 3D (not 2D planar gensteps)
            u1 = rng(); 
            cosTheta = u1 ; 
            sinTheta = sqrtf(1.0-u1*u1);

            float4 ori ; // copy position from genstep, historically has been origin   
            ori.x = gs.q1.f.x ;
            ori.y = gs.q1.f.y ;
            ori.z = gs.q1.f.z ;
            ori.w = 1.f ; // <-- dont copy the "float" gsid 

            float4 dir ; 
            SetGridPlaneDirection(dir, gridaxes, cosPhi, sinPhi, cosTheta, sinTheta );  

            // transform photon position and direction into the genstep frame
            qt.right_multiply_inplace( ori, 1.f );  // position 
            qt.right_multiply_inplace( dir, 0.f );  // direction

            unsigned char ucj = (j < 255 ? j : 255 ) ;  // photon index local to the genstep
            gsid.c4.w = ucj ;     // setting C4U union element to change gsid.u 

            unsigned identity = gsid.u ;   // include photon index IW with the genstep coordinate

            quad4& p = simtrace[count] ; 
            p.zero(); 

            if(simtrace_layout)  // follow layout of sevent::add_simtrace
            {
                p.q0.f = {0.f,0.f,0.f,0.f} ;   // intersect normal and distance
                p.q1.f = {0.f,0.f,0.f,0.f} ;   // intersect position
                p.q2.f = ori ;                 // initial position
                p.q3.f = dir ;                 // initial direction 
                p.q3.u.w = identity ;   
            } 
            else
            {
                p.q0.f = ori ; 
                p.q1.f = dir ; 
                p.q2.f = {0.f,0.f,0.f,0.f} ;
                p.q3.f = {0.f,0.f,0.f,0.f} ;
                p.q3.u.w = identity ;   
            }
            count += 1 ; 
        }
    }
    LOG(LEVEL) << " simtrace.size " << simtrace.size() ; 
}



/**
SFrameGenstep::SetGridPlaneDirection
----------------------------------

The cosTheta and sinTheta arguments are only used for the 3D gridaxes == XYZ

**/

void SFrameGenstep::SetGridPlaneDirection( float4& dir, int gridaxes, double cosPhi, double sinPhi, double cosTheta, double sinTheta )
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


#include "SStr.hh"
#include "SPath.hh"
#include "SSys.hh"
#include "PLOG.hh"

#include "scuda.h"
#include "sqat4.h"
#include "SEvent.hh"
#include "NP.hh"

#include "CSGFoundry.h"
#include "CSGGenstep.h"

//#include "SCenterExtentGenstep.hh"

/**
TODO: adopt SCenterExtentGenstep.hh
**/


const plog::Severity CSGGenstep::LEVEL = PLOG::EnvLevel("CSGGenstep", "DEBUG"); 

CSGGenstep::CSGGenstep( const CSGFoundry* foundry_ )
    :
    foundry(foundry_), 
    gridscale(SSys::getenvfloat("GRIDSCALE", 1.0 )), 
    moi(nullptr),
    midx(-1),
    mord(-1),
    iidx(-1),
    ce(make_float4(0.f, 0.f, 0.f, 100.f)),
    m2w(qat4::identity()),
    w2m(qat4::identity()),
    geotran(nullptr),
    gs(nullptr),
    pp(nullptr)
{
    init(); 
}

void CSGGenstep::init()
{
    SSys::getenvintvec("CEGS", cegs, ':', "5:0:5:1000" ); 
    // expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 
}

/**
CSGGenstep::create
--------------------

moi
   string identifying piece of geometry mName:mOrdinal:mInstance eg Hama:0:1000
   ordinal is to pick between global geom and instance between instanced 

ce_offset
   typically false for instanced geometry and true for global  

   this is kinda a noddy way of getting a transform 
   so this should be false when using tangential transforms 

ce_scale
   typically true for old style global  

TODO: eliminate ce_offset/ce_scale by using transform approach always


1. identify piece of geometry from moi
2. get location and transform for the geometry
3. configure grid to probe the geometry
4. create grid of gensteps 

**/

void CSGGenstep::create(const char* moi_, bool ce_offset, bool ce_scale )
{
    moi = strdup(moi_); 

    LOG(info) << " moi " << moi << " ce_offset " << ce_offset << " ce_scale " << ce_scale ; 

    if( strcmp(moi, "FAKE") == 0 ) 
    {
        std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
        gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ;
    }
    else
    {
        locate(moi); 
        configure_grid(); 
        gs = SEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale ); 
    }

    gs->set_meta<std::string>("moi", moi ); 
    gs->set_meta<int>("midx", midx); 
    gs->set_meta<int>("mord", mord); 
    gs->set_meta<int>("iidx", iidx); 
}


/**
CSGGenstep::locate
-----------------------

1. parseMoi string giving midx:mord:iidx 
2. get the *ce* and instance transform (populating qat4 members *m2w* *w2m*) 
   from the midx:mord:iidx using CSGTarget::getCenterExtent 
   which branches depending on iidx:

   iidx==-1 CSGTarget::getLocalCenterExtent 
   iidx>-1  CSGTarget::getGlobalCenterExtent 

3. form geotran from the instance transform  

Using CSGGenstepTest observed that with global non-instanced geometry 
are just using the identity transform from the single global "instance". 
Have added experimental way to use the tangential rtp transforms.


**/

void CSGGenstep::locate(const char* moi_)
{
    moi = strdup(moi_) ; 

    foundry->parseMOI(midx, mord, iidx, moi );  

    LOG(info) << " moi " << moi << " midx " << midx << " mord " << mord << " iidx " << iidx ;   
    if( midx == -1 )
    {    
        LOG(fatal)
            << " failed CSGFoundry::parseMOI for moi [" << moi << "]" 
            ;
        return ; 
    }


    int rc = foundry->getCenterExtent(ce, midx, mord, iidx, m2w, w2m ) ;

    LOG(info) << " rc " << rc << " MOI.ce (" 
              << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;           

    LOG(info) << "m2w" << *m2w ;
    LOG(info) << "w2m" << *w2m ;
 
    geotran = Tran<double>::FromPair( m2w, w2m, 1e-6 );    // Tran from stran.h 

    //geotran = Tran<double>::ConvertToTran( qt );    // Tran from stran.h 
    // matrix gets inverted by Tran<double>

    //override_locate(); 
}

void CSGGenstep::override_locate()
{
    std::vector<int> override_ce ; 
    SSys::getenvintvec("CXS_OVERRIDE_CE",  override_ce, ':', "0:0:0:0" ); 

    if( override_ce.size() == 4 && override_ce[3] > 0 )
    {
        ce.x = float(override_ce[0]); 
        ce.y = float(override_ce[1]); 
        ce.z = float(override_ce[2]); 
        ce.w = float(override_ce[3]); 
        LOG(info) << "override the MOI.ce with CXS_OVERRIDE_CE (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;  
    } 
}


/**
CSGGenstep::configure_grid
----------------------------

Config grid to probe the geometry 

**/

void CSGGenstep::configure_grid()
{
    SEvent::StandardizeCEGS(ce, cegs, gridscale ); 
    assert( cegs.size() == 7 ); 

    float3 mn ; 
    float3 mx ; 

    bool ce_offset_bb = true ; 
    SEvent::GetBoundingBox( mn, mx, ce, cegs, gridscale, ce_offset_bb ); 

    LOG(info) 
        << " ce_offset_bb " << ce_offset_bb
        << " mn " << mn 
        << " mx " << mx
        ; 
}

void CSGGenstep::generate_photons_cpu()
{
    pp = SEvent::GenerateCenterExtentGenstepsPhotons_( gs );  
    std::cout 
         << "gs " << gs->sstr() 
         << "pp " << pp->sstr() 
         << std::endl 
         ;
}

void CSGGenstep::save(const char* basedir) const 
{
    enum { e_dirpath = 2 } ; 
    const char* base = SPath::Resolve(basedir, moi, e_dirpath ); 
    LOG(info) << " save to " << base ; 
    if(gs) gs->save(base, "gs.npy"); 
    if(pp) pp->save(base, "pp.npy"); 
}



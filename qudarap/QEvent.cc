#include <cuda_runtime.h>

#include "SPath.hh"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"

#include "NP.hh"
#include "PLOG.hh"

#include "OpticksGenstep.h"

#include "QEvent.hh"
#include "QBuf.hh"
#include "QBuf.hh"
#include "QSeed.hh"
#include "QU.hh"

#include "qevent.h"

template struct QBuf<quad6> ; 

const plog::Severity QEvent::LEVEL = PLOG::EnvLevel("QEvent", "DEBUG"); 
const QEvent* QEvent::INSTANCE = nullptr ; 
const QEvent* QEvent::Get(){ return INSTANCE ; }


NP* QEvent::MakeGensteps(const std::vector<quad6>& gs ) // static 
{
    assert( gs.size() > 0); 
    NP* a = NP::Make<float>( gs.size(), 6, 4 ); 
    a->read2<float>( (float*)gs.data() ); 
    return a ; 
}



/**
QEvent::StandardizeCEGS
--------------------------

The cegs vector configures a grid. 
Symmetric and offset grid input configs are supported using 
vectors of length 4 and 7. 

This method standardizes the specification 
into an absolute index form which is used by 
QEvent::MakeCenterExtentGensteps

nx:ny:nz:num_photons
     symmetric grid -nx:nx, -ny:ny, -nz:nz  

nx:ny:nz:dx:dy:dz:num_photons
     offset grid -nx+dx:nx+dx, -ny+dy:ny+dy, -nz+dz:nz+dz  

ix0:iy0:iz0:ix1:iy1:iz1:num_photons 
     standardized absolute form of grid specification 

**/

void QEvent::StandardizeCEGS( const float4& ce, std::vector<int>& cegs, float gridscale ) // static 
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

    float x0 = float(ix0)*gridscale*ce.w ; 
    float x1 = float(ix1)*gridscale*ce.w ; 
    float y0 = float(iy0)*gridscale*ce.w ; 
    float y1 = float(iy1)*gridscale*ce.w ; 
    float z0 = float(iz0)*gridscale*ce.w ; 
    float z1 = float(iz1)*gridscale*ce.w ; 

    LOG(info) 
        << " CXS_CEGS "
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
QEvent::MakeCenterExtentGensteps
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



NP* QEvent::MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran ) // static
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

    gs.q0.i.x = OpticksGenstep_TORCH ;  
    gs.q0.i.y = 0 ; // could plant enum for XZ planar etc.. here 
    gs.q0.i.z = 0 ; // 
    gs.q0.i.w = photons_per_genstep ; 

    gs.q1.f.x = 0.f ;  // local frame position : currently origin, same for all gensteps : only the transform is changed   
    gs.q1.f.y = 0.f ;  
    gs.q1.f.z = 0.f ;   
    gs.q1.f.w = 1.f ; 

    for(int ix=ix0 ; ix < ix1+1 ; ix++ )
    for(int iy=iy0 ; iy < iy1+1 ; iy++ )
    for(int iz=iz0 ; iz < iz1+1 ; iz++ )
    {
        LOG(LEVEL) << " ix " << ix << " iy " << iy << " iz " << iz  ; 
        
        double tx = double(ix)*gridscale*ce.w ; 
        double ty = double(iy)*gridscale*ce.w ; 
        double tz = double(iz)*gridscale*ce.w ; 
        const Tran<double>* local_translate = Tran<double>::make_translate( tx, ty, tz );   // TODO: stack not heap ?

        bool reverse = false ; 
        const Tran<double>* transform = Tran<double>::product( geotran, local_translate, reverse );  

        qat4* qc = Tran<double>::ConvertFrom( transform->t ) ; 

        qc->write(gs);                    // copy qc into gs.q2,q3,q4,q5

        gensteps.push_back(gs); 
    }
    LOG(LEVEL) << " gensteps.size " << gensteps.size() ; 

    return MakeGensteps(gensteps); 
}








NP* QEvent::MakeCountGensteps() // static 
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    return MakeCountGensteps(photon_counts_per_genstep); 
}
NP* QEvent::MakeCountGensteps(const std::vector<int>& counts) // static 
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






QEvent::QEvent()
    :
    evt(new qevent),
    d_evt(QU::device_alloc<qevent>(1)),
    genstep(nullptr),
    seed(nullptr),
    meta(nullptr)
{
    INSTANCE = this ; 
}

void QEvent::setMeta(const char* meta_)
{
    meta = meta_ ; 
} 

void QEvent::setGensteps(const NP* gs_) 
{ 
    gs = gs_ ; 

    assert( gs->uifc == 'f' && gs->ebyte == 4 ); 
    assert( gs->has_shape(-1, 6, 4) ); 
    unsigned num_gs = gs->shape[0] ; 
    LOG(info) << " num_gs " << num_gs ; 

    QBuf<float>* dgs = QBuf<float>::Upload( gs );   // TODO: this is allocating every time, better to resize to avoid GPU leaking  
    setGensteps(dgs); 
}

void QEvent::setGensteps(QBuf<float>* dgs) // QBuf::ptr references already uploaded gensteps
{
    genstep = dgs ; 
    seed = QSeed::CreatePhotonSeeds(genstep);
    if(!seed) LOG(fatal) << " FAILED to QSeed::CreatePhotonSeeds : problem with gensteps ? " ; 
    assert( seed ); 

    evt->genstep = (quad6*)genstep->ptr ; 
    evt->seed = seed->ptr ; 
    evt->num_photon = seed->num_items ; 
    evt->photon = QU::device_alloc<quad4>(evt->num_photon) ; 

    QU::copy_host_to_device<qevent>(d_evt, evt, 1 );  
}


void QEvent::downloadPhoton( std::vector<quad4>& photon )
{
    photon.resize(evt->num_photon); 
    QU::copy_device_to_host_and_free<quad4>( photon.data(), evt->photon, evt->num_photon ); 
}

void QEvent::savePhoton( const char* dir_, const char* name )
{
    int create_dirs = 2 ;  // 2:dirpath 
    const char* dir = SPath::Resolve(dir_, create_dirs); 

    LOG(info) << dir ; 
    std::vector<quad4> photon ; 
    downloadPhoton(photon); 
    NP::Write( dir, name,  (float*)photon.data(), photon.size(), 4, 4  );
}

void QEvent::saveGenstep( const char* dir_, const char* name)
{
    if(!gs) return ; 
    int create_dirs = 1 ;  // 1:filepath 
    const char* path = SPath::Resolve(dir_, name, create_dirs); 
    gs->save(path); 
}

void QEvent::saveMeta( const char* dir_, const char* name)
{     
    if(!meta) return ; 
    int create_dirs = 1 ;  // 1:filepath 
    const char* path = SPath::Resolve(dir_, name, create_dirs); 
    NP::WriteString(path, meta ); 
}

std::string QEvent::desc() const
{
    std::stringstream ss ; 
    ss 
        << " genstep " << ( genstep ? genstep->desc() : "-" ) 
        << " seed " << ( seed ? seed->desc() : "-" ) 
        ;
    return ss.str(); 
}


qevent* QEvent::getDevicePtr() const
{
    return d_evt ; 
}
unsigned QEvent::getNumPhotons() const
{
    return evt->num_photon ; 
}


extern "C" void QEvent_checkEvt(dim3 numBlocks, dim3 threadsPerBlock, qevent* evt, unsigned width, unsigned height ) ; 

void QEvent::checkEvt() 
{ 
    unsigned width = getNumPhotons() ; 
    unsigned height = 1 ; 
    LOG(info) << " width " << width << " height " << height ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 
 
    assert( d_evt ); 
    QEvent_checkEvt(numBlocks, threadsPerBlock, d_evt, width, height );   
}




#include "scuda.h"
#include "squad.h"
#include "stran.h"

#include "SSys.hh"
#include "SPath.hh"
#include "SEvent.hh"
#include "SCenterExtentGenstep.hh"
#include "NP.hh"

#include "PLOG.hh"

void SCenterExtentGenstep::save(const char* dir) const
{
    LOG(info) << "[ ii.size " << ii.size() ;
    NP* isect = NP::Make<float>(ii.size(), 4, 4);
    LOG(info) << isect->sstr() ;
    isect->read<float>((float*)ii.data());
    isect->save(dir, "isect.npy");
    gs->save(dir, "gs.npy" ); 

    int create_dirs = 1 ; // 1:filepath
    const char* peta_path = SPath::Resolve(dir, "peta.npy", create_dirs) ;
    NP::Write(peta_path, (float*)(&peta->q0.f.x), 1, 4, 4 );

    LOG(info) << "]" ; 
}



template<typename T> void SCenterExtentGenstep::set_meta(const char* key, T value ) 
{
    assert(gs); 
    gs->set_meta<T>(key, value) ; 
}


SCenterExtentGenstep::SCenterExtentGenstep()
    :
    gs(nullptr),
    gridscale(SSys::getenvfloat("GRIDSCALE", 1.0 )),
    peta(new quad4)
{
    init(); 
} 

void SCenterExtentGenstep::init()
{
    peta->zero(); 

    LOG(info) << "[ gridscale " << gridscale  ;

    ce = make_float4(0.f, 0.f, 0.f, 100.f );

    SSys::getenvintvec("CXS_CEGS", cegs, ':', "16:0:9:10" );
    // expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 

    SEvent::StandardizeCEGS(ce, cegs, gridscale );
    assert( cegs.size() == 7 );

    int ix0 = cegs[0] ;
    int ix1 = cegs[1] ;
    int iy0 = cegs[2] ;
    int iy1 = cegs[3] ;
    int iz0 = cegs[4] ;
    int iz1 = cegs[5] ;
    int photons_per_genstep = cegs[6] ;
    int zero = 0 ;

    nx = (ix1 - ix0)/2 ;
    ny = (iy1 - iy0)/2 ;
    nz = (iz1 - iz0)/2 ;
    int gridaxes = SEvent::GridAxes(nx, ny, nz);

    LOG(info)
        << " nx " << nx
        << " ny " << ny
        << " nz " << nz
        << " GridAxes " << gridaxes
        << " GridAxesName " << SEvent::GridAxesName(gridaxes)
        ;


    peta->q0.i.x = ix0 ;
    peta->q0.i.y = ix1 ;
    peta->q0.i.z = iy0 ;
    peta->q0.i.w = iy1 ;

    peta->q1.i.x = iz0 ;
    peta->q1.i.y = iz1 ;
    peta->q1.i.z = photons_per_genstep ;
    peta->q1.i.w = zero ;



    SSys::getenvintvec("CXS_OVERRIDE_CE",  override_ce, ':', "0:0:0:0" );

    const Tran<double>* geotran = Tran<double>::make_identity();

    if( override_ce.size() == 4 && override_ce[3] > 0 )
    {
        ce.x = float(override_ce[0]);
        ce.y = float(override_ce[1]);
        ce.z = float(override_ce[2]);
        ce.w = float(override_ce[3]);
        LOG(info) << "override ce with CXS_OVERRIDE_CE (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;
    }

    peta->q2.f.x = ce.x ;   // moved from q1
    peta->q2.f.y = ce.y ;
    peta->q2.f.z = ce.z ;
    peta->q2.f.w = ce.w ;

    bool ce_offset = false ;
    bool ce_scale = true ;

    gs = SEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );

    SEvent::GenerateCenterExtentGenstepsPhotons( pp, gs );

    LOG(info) << "]" ;
}





const char* SCenterExtentGenstep::desc() const 
{
    std::stringstream ss ; 
    ss << " CEGS (" ; 
    for(unsigned i=0 ; i < cegs.size() ; i++ ) ss << cegs[i] << " " ; 
    ss << ")" ; 
    ss << " nx " << nx ; 
    ss << " ny " << ny ; 
    ss << " nz " << nz ; 
    ss << " GRIDSCALE " << gridscale ; 
    ss << " CE (" 
       << ce.x << " " 
       << ce.y << " " 
       << ce.z << " " 
       << ce.w 
       << ") " 
       ;   

    ss << " gs " << gs->sstr() ; 
    ss << " pp " << pp.size() ; 
    ss << " ii " << ii.size() ; 

    std::string s = ss.str(); 
    return strdup(s.c_str()); 
}


template void     SCenterExtentGenstep::set_meta<int>(const char*, int ); 
template void     SCenterExtentGenstep::set_meta<unsigned>(const char*, unsigned ); 
template void     SCenterExtentGenstep::set_meta<float>(const char*, float ); 
template void     SCenterExtentGenstep::set_meta<double>(const char*, double ); 
template void     SCenterExtentGenstep::set_meta<std::string>(const char*, std::string ); 




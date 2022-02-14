#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "ssincos.h"


#include "SPath.hh"
#include "SEvent.hh"
#include "SRng.hh"

#include "NP.hh"

const char* BASE = "$TMP/sysrap/SEventTest" ; 

const NP* test_MakeCountGensteps()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );  
    gs->save(fold, "cngs.npy"); 

    return gs ; 
}


const NP* test_MakeCenterExtentGensteps(int nx, int ny, int nz) 
{
    LOG(info); 

    float4 ce = make_float4( 1.f, 2.f, 3.f, 100.f );  

    LOG(info) << " ce " << ce ; 


    std::vector<int> cegs = {{nx, ny, nz, 10 }} ; 
    float gridscale = 1.f ; 

    SEvent::StandardizeCEGS(ce, cegs, gridscale );

    float3 mn ; 
    float3 mx ; 

    for( int i=0 ; i < 2 ; i++)
    {
        bool ce_offset_bb = i == 1 ; 
        SEvent::GetBoundingBox(mn, mx, ce, cegs, gridscale, ce_offset_bb ); 
        std::cout 
            << " ce_offset_bb " << ce_offset_bb
            << " mn " << mn 
            << " mx " << mx
            << std::endl
            ;  
     }


    //bool rot = false ;  // 45 degress around Z   OR identity 
    //const Tran<float>* tr = rot ? Tran<float>::make_rotate( 0., 0., 1., 45. ) : Tran<float>::make_identity() ;
    //std::cout << " tr " << *tr << std::endl ; 
    //qat4* qt_ptr = new qat4( tr->tdata() ); 

    int idx = 1 ; 

    const char* str_0 = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    const char* str_1 = "(100,100,100)" ; 
    const char* str = nullptr ; 
    switch(idx)
    {
        case 0: str = str_0 ; break ; 
        case 1: str = str_1 ; break ; 
    }

    qat4* qt = qat4::from_string(str); 
    const Tran<double>* geotran = Tran<double>::ConvertToTran(qt); 
    bool ce_offset = false ; 
    bool ce_scale = true ; 

    const NP* gs = SEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );  

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );
    gs->save(fold, "cegs.npy");

    return gs ;
}




void test_GenerateCenterExtentGensteps_0( const NP* gsa )
{   
    LOG(info); 

    float gridscale = 1.f ; // not usually used
    std::vector<quad4> pp ;
    SEvent::GenerateCenterExtentGenstepsPhotons( pp, gsa, gridscale ); 
    NP* ppa = NP::Make<float>( pp.size(), 4, 4 ); 
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );
   
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );
    std::cout << "ppa " << ppa->sstr() << std::endl ;
    ppa->save(fold, "ppa.npy"); 
}

void test_GenerateCenterExtentGensteps_1( const NP* gsa )
{   
    LOG(info); 
    float gridscale = 1.f ; // not usually used
    NP* ppa = SEvent::GenerateCenterExtentGenstepsPhotons_( gsa, gridscale ); 
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );
    std::cout << "ppa " << ppa->sstr() << std::endl ;
    ppa->save(fold, "ppa.npy"); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const NP* gs = test_MakeCountGensteps() ; 
    const NP* gs0 = test_MakeCenterExtentGensteps(3, 0, 3) ;
    assert( gs0 ); 
    gs0->dump(); 

    //test_GenerateCenterExtentGensteps_0(gs0); 
    test_GenerateCenterExtentGensteps_1(gs0); 

    return 0 ; 
}           



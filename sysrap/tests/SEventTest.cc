#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "ssincos.h"


#include "SPath.hh"
#include "SEvent.hh"
#include "SFrameGenstep.hh"
#include "SCenterExtentGenstep.hh"
#include "SRng.hh"

#include "NP.hh"

const char* BASE = "$TMP/sysrap/SEventTest" ; 

const NP* test_MakeCountGensteps()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    int x_total = 0 ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep, &x_total ) ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );  
    gs->save(fold, "cngs.npy"); 

    return gs ; 
}


const Tran<double>* GetTestTransform(int idx)
{
    const Tran<double>* geotran = nullptr ;  

    if( idx == 0 )
    {
        geotran = Tran<double>::make_identity() ; 
    }
    else if( idx < 10 )
    {
        const char* str_1 = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
        const char* str_2 = "(100,100,100)" ; 
        const char* str_3 = "(0,0,0)" ; 
        const char* str = nullptr ; 
        switch(idx)
        {
            case 1: str = str_1 ; break ; 
            case 2: str = str_2 ; break ; 
            case 3: str = str_3 ; break ; 
        }

        qat4* qt = qat4::from_string(str); 
        geotran = Tran<double>::ConvertToTran(qt); 
    }
    else if( idx == 10 )
    {
        geotran = Tran<double>::make_rotate( 0., 0., 1., 45. ) ;
    }
    return geotran ; 
}
 


const NP* test_MakeCenterExtentGensteps(int nx, int ny, int nz, const float4* ce_ ) 
{
    LOG(info); 
    float4 ce( ce_ ? *ce_ :  make_float4( 1.f, 2.f, 3.f, 100.f ));  

    LOG(info) << " ce " << ce ; 

    std::vector<int> cegs = {{nx, ny, nz, 10 }} ; 
    float gridscale = 1.f ; 

    SFrameGenstep::StandardizeCEGS(ce, cegs, gridscale );

    bool ce_scale = true ; 
    float3 offset = make_float3(0.f, 0.f, 0.f) ; 
    std::vector<float3> ce_offset ; 
    ce_offset.push_back(offset);  

    const Tran<double>* geotran = GetTestTransform(0) ; 

    const NP* gs = SFrameGenstep::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );  

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
    SFrameGenstep::GenerateCenterExtentGenstepsPhotons( pp, gsa, gridscale ); 
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
    NP* ppa = SFrameGenstep::GenerateCenterExtentGenstepsPhotons_( gsa, gridscale ); 
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs );
    LOG(info) << "ppa " << ppa->sstr() << " saving ppa.npy to " << fold  ;
    ppa->save(fold, "ppa.npy"); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float4 ce ; 
    qvals( ce, "CE", "500,0,0,100" ); 
    LOG(info) << " ce " << ce ; 

    //const NP* gs = test_MakeCountGensteps() ; 
    const NP* gs0 = test_MakeCenterExtentGensteps(3, 0, 3, &ce ) ;
    assert( gs0 ); 
    gs0->dump(0,gs0->shape[0],4,6); 

    //test_GenerateCenterExtentGensteps_0(gs0); 
    test_GenerateCenterExtentGensteps_1(gs0); 

    return 0 ; 
}           



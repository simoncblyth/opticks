/**
SEventTest.cc
===============

TEST=MakeCountGenstep SEventTest


**/


#include "OPTICKS_LOG.hh"

#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "stran.h"
#include "ssincos.h"

#include "SEvent.hh"
#include "SFrameGenstep.hh"
#include "SCenterExtentGenstep.hh"
#include "SRng.hh"

#include "NP.hh"

const char* FOLD = "${FOLD:-$TMP/sysrap/SEventTest}" ;

struct SEventTest
{
    static const char* TEST ;

    static int Resolve();
    static int MakeCountGenstep();
    static int MakeTorchGenstep();

    static const Tran<double>* GetTestTransform(int idx);

    static int       MakeCenterExtentGenstep_();
    static const NP* MakeCenterExtentGenstep(int nx, int ny, int nz );

    static int GenerateCEG0_();
    static int GenerateCEG0( const NP* gsa );

    static int GenerateCEG1_();
    static int GenerateCEG1( const NP* gsa );

    static int Main();
};

const char* SEventTest::TEST = ssys::getenvvar("TEST", "Resolve");

int SEventTest::Resolve()
{
   const char* path = spath::Resolve(FOLD);
   std::cout
        << "SEventTest::Resolve\n"
        << " FOLD [" << FOLD << "]\n"
        << " path [" << ( path ? path : "-" ) << "]\n"
        ;
   return 0 ;
}

int SEventTest::MakeCountGenstep()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    int x_total = 0 ;
    const NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_total ) ;
    gs->save(FOLD, "cngs.npy");

    std::cout
        << "SEventTest::MakeCountGenstep"
        << " x_total " << x_total
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    return 0 ;
}

int SEventTest::MakeTorchGenstep()
{
    const char* path = spath::Resolve(FOLD, "torch.npy");
    const NP* gs = SEvent::MakeTorchGenstep() ;
    gs->save(path);

    std::cout
        << "SEventTest::MakeTorchGenstep"
        << " path " << ( path ? path : "-" )
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    return 0 ;
}







const Tran<double>* SEventTest::GetTestTransform(int idx)
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


int SEventTest::MakeCenterExtentGenstep_()
{
    const NP* gs = MakeCenterExtentGenstep( 2, 2, 2);
    std::cout << " gs " << ( gs ? gs->sstr() : "-" ) << "\n" ;
    return 0 ;
}

const NP* SEventTest::MakeCenterExtentGenstep(int nx, int ny, int nz )
{
    float4 ce ;
    qvals( ce, "CE", "500,0,0,100" );
    LOG(info) << " ce " << ce ;

    std::vector<int> cegs = {{nx, ny, nz, 10 }} ;
    float gridscale = 1.f ;

    SFrameGenstep::StandardizeCEGS(cegs);

    bool ce_scale = true ;
    float3 offset = make_float3(0.f, 0.f, 0.f) ;
    std::vector<float3> ce_offset ;
    ce_offset.push_back(offset);

    const Tran<double>* geotran = GetTestTransform(0) ;

    const NP* gs = SFrameGenstep::MakeCenterExtentGenstep(ce, cegs, gridscale, geotran, ce_offset, ce_scale, nullptr );

    const char* path = spath::Resolve(FOLD, "cegs.npy");
    gs->save(path);

    std::cout
         << "SEventTest::MakeCenterExtentGenstep"
         << " path [" << ( path ? path : "-" ) << "]\n"
         << " gs " << ( gs ? gs->sstr() : "-" )
         << "\n"
         ;

    return gs ;
}


int SEventTest::GenerateCEG0_()
{
    const NP* gs = MakeCenterExtentGenstep(3, 0, 3) ;
    assert( gs );
    gs->dump(0,gs->shape[0],4,6);

    return GenerateCEG0(gs);
}

int SEventTest::GenerateCEG0( const NP* gsa )
{
    LOG(info);

    float gridscale = 1.f ; // not usually used
    std::vector<quad4> pp ;
    SFrameGenstep::GenerateCenterExtentGenstepPhotons( pp, gsa, gridscale );
    NP* ppa = NP::Make<float>( pp.size(), 4, 4 );
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );

    std::cout << "ppa " << ppa->sstr() << std::endl ;
    ppa->save(FOLD, "ppa.npy");

    return 0 ;
}

int SEventTest::GenerateCEG1_()
{
    const NP* gs = MakeCenterExtentGenstep(3, 0, 3) ;
    assert( gs );
    gs->dump(0,gs->shape[0],4,6);

    return GenerateCEG1(gs);
}

int SEventTest::GenerateCEG1( const NP* gsa )
{
    LOG(info);
    float gridscale = 1.f ; // not usually used
    NP* ppa = SFrameGenstep::GenerateCenterExtentGenstepPhotons_( gsa, gridscale );
    LOG(info) << "ppa " << ppa->sstr() << " saving ppa.npy to " << FOLD  ;
    ppa->save(FOLD, "ppa.npy");
    return 0 ;
}

int SEventTest::Main()
{
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    std::cout << "SEventTest::Main TEST " << TEST << "\n" ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST, "Resolve"))                 rc += Resolve();
    if(ALL||0==strcmp(TEST, "MakeCountGenstep"))        rc += MakeCountGenstep();
    if(ALL||0==strcmp(TEST, "MakeTorchGenstep"))        rc += MakeTorchGenstep();

    if(ALL||0==strcmp(TEST, "MakeCenterExtentGenstep")) rc += MakeCenterExtentGenstep_();
    if(ALL||0==strcmp(TEST, "GenerateCEG0"))            rc += GenerateCEG0_();
    if(ALL||0==strcmp(TEST, "GenerateCEG1"))            rc += GenerateCEG1_();

    std::cout << "SEventTest::Main TEST " << TEST << " rc " << rc << "\n" ;
    return rc ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    return SEventTest::Main() ;
}



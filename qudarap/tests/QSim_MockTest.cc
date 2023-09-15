/**
QSim_MockTest.cc : CPU tests of QSim.hh/qsim.h CUDA code using MOCK_CURAND mocking 
=======================================================================================


Testing GPU code on CPU requires mocking 
of CUDA API including:

1. tex2D lookups 
2. curand random generation
3. erfcinvf : inverse complementary error function 

There are now lots of examples of curand mocking, 
search for MOCK_CURAND, MOCK_CUDA. See::

    sysrap/s_mock_curand.h 
    sysrap/scurand.h 

Mocking tex2D lookups is not so common. See::

    sysrap/s_mock_texture.h 
    sysrap/stexture.h 

and search for MOCK_TEXTURE, MOCK_CUDA. 


HMM: QSim.hh not very amenable to standalone use because the boatload of 
headers that come with it : so operating at lower level.

Standalone compile and run with::

   ./QSim_MockTest.sh 

**/

#include "NPFold.h"

#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "smath.h"    // includes s_mock_erfinvf.h when MOCK_CUDA is defined
#include "squad.h"
#include "srec.h"
#include "stag.h"
#include "sflow.h"
#include "sphoton.h"
#include "sstate.h"
#include "scerenkov.h"

#include "scurand.h"    // includes s_mock_curand.h when MOCK_CURAND OR MOCK_CUDA defined 
#include "stexture.h"   // includes s_mock_texture.h when MOCK_TEXTURE OR MOCK_CUDA defined 

#include "SPMT.h"
#include "SBnd.h"
#include "OpticksPhoton.hh"

#include "QBase.hh"
#include "QPMT.hh"
#include "QBnd.hh"
#include "QOptical.hh"

#include "qpmt.h"
#include "qbnd.h"
#include "qsim.h"

struct QSim_MockTest
{
    static constexpr const char* BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard" ; 
    static constexpr const char* BND = 
    "Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum" ;

    const char* FOLD ; 
    const char* CHECK ; 
    curandStateXORWOW rng ; 

    const NP* optical ; 
    const NP* bnd ; 

    const QBase*    q_base ; 
    const QOptical* q_optical ; 
    const QBnd*     q_bnd ; 

    const SBnd*     s_bnd ; 
    int   boundary ; 

    const NPFold* jpmt ; 
    const QPMT<float>* q_pmt ; 
    qsim* sim ; 
    const int num ;   
    const int PIDX ; 
    NP*  a ; 

    QSim_MockTest(); 

    void init(); 
    void init_bnd(); 
    std::string desc() const; 

    void generate_photon_dummy(); 
    void uniform_sphere();

    void propagate_at_boundary_manual();

    void setup_prd(    quad2& prd ); 
    void setup_photon( sphoton& p );

#ifdef WITH_CUSTOM4
    void propagate_at_surface_CustomART_manual();   
#endif

    void fill_state(); 
    void propagate_at_boundary();
    void propagate();
    void SmearNormal(int chk, float value); 
    void SmearNormal_SigmaAlpha_one(); 

    void run(); 
};

inline QSim_MockTest::QSim_MockTest()
    :
    FOLD(ssys::getenvvar("FOLD")),
    CHECK(spath::Basename(FOLD)),
    rng(1u),
    optical(NP::Load(BASE, "optical.npy")),
    bnd(    NP::Load(BASE, "bnd.npy")),
    q_base( new QBase ),
    q_optical(optical ? new QOptical(optical) : nullptr), 
    q_bnd(    bnd     ? new QBnd(bnd)         : nullptr), 
    s_bnd(    bnd     ? new SBnd(bnd)         : nullptr),
    boundary(s_bnd ? s_bnd->getBoundaryIndex(BND) : -1),
    jpmt(SPMT::Serialize()),
    q_pmt( jpmt ? new QPMT<float>( jpmt ) : nullptr),  
    sim(new qsim),
    num(ssys::getenvint("NUM",1000)),
    PIDX(ssys::getenvint("PIDX",-100)),
    a(nullptr)
{
    init(); 
}


inline void QSim_MockTest::init()
{
    init_bnd(); 

    assert( q_pmt ); 
    //rng.set_fake(0.); // 0/1:forces transmit/reflect 

    sim->base = q_base ? q_base->d_base : nullptr ;
    sim->pmt = q_pmt->d_pmt ; 
}


inline void QSim_MockTest::init_bnd()
{
    assert( bnd ); 
    bool have_boundary = boundary > -1 ; 
    if(!have_boundary) std::cerr 
        << "QSim_MockTest::init_bnd"
        << std::endl 
        << " FATAL FAILED TO LOOKUP boundary " << boundary
        << std::endl 
        << " BND " << BND 
        << std::endl 
        << " NO BOUNDARY WITH THAT SPEC PRESENT WITHIN LOADED GEOMETRY "
        << std::endl 
        << " BASE " << BASE
        << std::endl 
        << " PROBABLY THE GEOM ENVVAR IS MIS-CONFIGURED : CHANGE IT USING GEOM BASH FUNCTION " 
        << std::endl
        ;
    assert( have_boundary ); 
    sim->bnd = q_bnd->d_qb ;  
}


inline std::string QSim_MockTest::desc() const
{
    std::stringstream ss ; 
    ss << "QSim_MockTest::desc" << std::endl 
       << " bnd " << ( bnd ? bnd->sstr() : "-" )
       << " optical " << ( optical ? optical->sstr() : "-" )
       << " boundary " << boundary << std::endl  
       << " s_bnd.getBoundarySpec " << ( s_bnd ? s_bnd->getBoundarySpec(boundary) : "-" )
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline void QSim_MockTest::generate_photon_dummy()
{
    sphoton p ; 
    p.zero(); 
    quad6 gs ; 
    gs.zero(); 
    unsigned photon_id = 0 ; 
    unsigned genstep_id = 0 ; 
    sim->generate_photon_dummy(p, rng, gs, photon_id, genstep_id ); 
}

inline void QSim_MockTest::uniform_sphere()
{
    for(int i=0 ; i < 10 ; i++)
    {
        float3 dir = sim->uniform_sphere(rng); 
        printf("//test_uniform_sphere dir (%10.4f %10.4f %10.4f) \n", dir.x, dir.y, dir.z ); 
    }
}


/**
QSim_MockTest::propagate_at_boundary_manual
---------------------------------------------

               n
           i   :   r     
            \  :  /
             \ : /
              \:/
          -----+-----------
                \
                 \
                  \
                   t

Consider mom is some direction, say +Z::

   (0, 0, 1)

There is a circle of vectors that are perpendicular 
to that mom, all in the XY plane::

   ( cos(phi), sin(phi), 0 )    phi 0->2pi 

Clearly the dot product if that and +Z is zero. 
     
**/
inline void QSim_MockTest::propagate_at_boundary_manual()
{
    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)) ; 

    std::cout 
        << " QSim_MockTest::propagate_at_boundary "
        << " mom " << mom 
        << std::endl 
        ;

    quad2 prd ; 
    setup_prd(prd) ; 

    sctx ctx ; 
    ctx.prd = &prd ; 

    sstate& s = ctx.s ; 
    s.material1.x = 1.0f ; 
    s.material2.x = 1.5f ; 
    // The two RINDEX are the only state needed for propagate_at_boundary  
    // TODO: get these in mocked up way from the bnd texture using the wavelength 


    sphoton& p = ctx.p ; 
    p.zero(); 
    p.wavelength = 440.f ; 

    unsigned flag = 0 ;  
    const int N = 16 ; 

    std::vector<sphoton> pp(N*2) ; 

    for(int i=0 ; i < N ; i++)
    {   
        float frac_twopi = float(i)/float(N)  ;   

        p.mom = mom ; 
        p.set_polarization(frac_twopi) ;  

        sphoton p0(p) ;  
        int ctrl = sim->propagate_at_boundary(flag, rng, ctx) ; 

        pp[i*2+0 ] = p0 ; 
        pp[i*2+1 ] = p ; 

        std::cout 
            << " flag " << OpticksPhoton::Flag(flag) 
            << " ctrl " <<  sflow::desc(ctrl) 
            << std::endl
            << " p0 " << p0.descDir()
            << std::endl
            << " p  " << p.descDir() 
            << std::endl
            ; 
     }

     NP* a = NP::Make<float>(N,2,4,4) ; 
     a->read2<float>( (float*)pp.data() ); 
     a->save("$FOLD/pp.npy");  
}


inline void QSim_MockTest::propagate_at_boundary()
{
    std::cout 
        << "QSim_MockTest::propagate_at_boundary"
        << std::endl 
        ;

    quad2 prd ; 
    setup_prd(prd) ; 

    sctx ctx ; 
    ctx.idx = 0 ; 
    ctx.prd = &prd ; 

    sphoton& p = ctx.p ; 
    setup_photon(p); 

    std::cout << "p0 " << p << std::endl ; 
   
    float cosTheta = -dot( p.mom, *prd.normal() ); 

    sim->bnd->fill_state(ctx.s, boundary, ctx.p.wavelength, cosTheta, ctx.idx );


    unsigned flag = 0 ;  
    int ctrl = sim->propagate_at_boundary(flag, rng, ctx) ; 

    std::cout 
        << " flag " << OpticksPhoton::Flag(flag) 
        << " ctrl " <<  sflow::desc(ctrl) 
        << std::endl
        ;

    std::cout << "p1 " << p << std::endl ; 
 
}


inline void QSim_MockTest::setup_prd( quad2& prd )
{
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 

    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;  
    prd.q0.f.w = 0.1f ;    // distance

    prd.q1.f.x = 0.5f ;  // lposcost : local position cosTheta of intersect 
    prd.q1.u.y = 0u ; 
    prd.q1.u.z = 1001 ;   // lpmtid : sensor_identifier (< 17612 )
    prd.q1.u.w = boundary ; 
}
inline void QSim_MockTest::setup_photon( sphoton& p)
{
    p.zero(); 
    p.pos = make_float3( 0.f, 0.f, 0.f );
    p.mom = normalize(make_float3(1.f, 0.f, -1.f));
    p.pol = normalize(make_float3(0.f, 1.f,  0.f));
    p.wavelength = 440.f ; 
}


#ifdef WITH_CUSTOM4

/**
QSim_MockTest::propagate_at_surface_CustomART_manual
-----------------------------------------------------

Version with manual sstate filling  

**/

inline void QSim_MockTest::propagate_at_surface_CustomART_manual()
{
    quad2 prd ; 
    setup_prd(prd) ; 

    sctx ctx ; 
    ctx.prd = &prd ; 

    sphoton& p = ctx.p ; 
    setup_photon(p); 

    // lpmtcat doesnt matter as Pyrex and Vacuum are same for all of them 
    float n0 = sim->pmt->get_lpmtcat_rindex_wl( 0, 0, 0, p.wavelength ); 
    float n3 = sim->pmt->get_lpmtcat_rindex_wl( 0, 3, 0, p.wavelength ); 

    std::cout 
        << "QSim_MockTest::propagate_at_surface_CustomART_manual "
        << " boundary " << boundary
        << " prd.q1.u.w " << prd.q1.u.w
        << " prd.q0.f:nrm " << prd.q0.f 
        << " p.mom " << p.mom 
        << " n0 " << std::fixed << std::setw(10) << std::setprecision(4) << n0 
        << " n3 " << std::fixed << std::setw(10) << std::setprecision(4) << n3 
        << std::endl 
        ;

    ctx.s.material1.x = n0 ;  // Pyrex RINDEX
    ctx.s.material2.x = n3 ;  // Vacuum RINDEX 

    unsigned flag = 0 ;  
    int ctrl = sim->propagate_at_surface_CustomART(flag, rng, ctx) ; 
 
    std::cout 
        << "QSim_MockTest::propagate_at_surface_CustomART_manual "
        << " flag " << flag << " : " << OpticksPhoton::Flag(flag) 
        << " ctrl " << ctrl << " : " << sflow::desc(ctrl)  
        << std::endl 
        ;
}
#endif 



inline void QSim_MockTest::fill_state()
{
    std::cout 
        << "QSim_MockTest::fill_state" 
        << " boundary " << boundary 
        << std::endl 
        ;

    sctx ctx ; 
    ctx.p.wavelength = 440.f  ; // nm 
    ctx.idx = 0 ; 

    for(float cosTheta=-1.f ; cosTheta <= 1.f ; cosTheta+= 2.f )
    {
        sim->bnd->fill_state(ctx.s, boundary, ctx.p.wavelength, cosTheta, ctx.idx );
        std::cout 
            << " cosTheta " << cosTheta 
            << std::endl 
            << " ctx.s "   
            << std::endl 
            << ctx.s  
            << std::endl
            ; 
    }
}


/**
QSim_MockTest::propagate
--------------------------

Does qsim::propagate_to_boundary and branches to the 
appropriate qsim::propagate_at_boundary method. 

**/

inline void QSim_MockTest::propagate()
{
    quad2 prd ; 
    setup_prd(prd) ; 

    sctx ctx ; 
    ctx.idx = 0 ; 
    ctx.prd = &prd ; 

    sphoton& p = ctx.p ; 
    setup_photon(p); 

    int bounce = 0 ; 

    sphoton p0(p) ; 
    sim->propagate(bounce, rng, ctx );
    sphoton p1(p) ; 

    std::cout << "p0 " << p0 << std::endl ; 
    std::cout << "p1 " << p1 << std::endl ; 
}

/**
QSim_MockTest::SmearNormal
---------------------------

This MOCK_CUDA test of qsim::SmearNormal_SigmaAlpha qsim::SmearNormal_Polish
is similar to sysrap/tests/S4OpBoundaryProcessTest.sh 

+------+-------------+
| chk  |  value      |
+======+=============+
|  0   | sigma_alpha |
+------+-------------+
|  1   | polish      |
+------+-------------+
**/

inline void QSim_MockTest::SmearNormal(int chk, float value)
{
    float3 direct = make_float3(0.f, 0.f, -1.f ); 
    float3 normal = make_float3(0.f, 0.f,  1.f ); 

    int ni = num ; 
    int nj = 4 ; 

    a = NP::Make<float>( ni, nj );  

    a->set_meta<std::string>("source", "QSim_MockTest.sh" ); 
    a->set_meta<std::string>("normal", scuda::serialize(normal) ); 
    a->set_meta<std::string>("direct", scuda::serialize(direct) ); 
    a->set_meta<float>("value", value );  
    a->set_meta<std::string>("valuename", chk == 0 ? "sigma_alpha" : "polish"  ); 
    a->names.push_back( chk == 0 ? "SmearNormal_SigmaAlpha" : "SmearNormal_Polish" ); 


    sctx ctx ; 


    float* aa = a->values<float>(); 
    for(int i=0 ; i < ni ; i++)
    {
        rng.setSequenceIndex(i); 
        ctx.idx = i ; 
        float3* smeared_normal = (float3*)(aa + i*nj + 0) ; 
        switch(chk)
        {
            case 0: sim->SmearNormal_SigmaAlpha(rng, smeared_normal, &direct, &normal, value, ctx ); break ; 
            case 1: sim->SmearNormal_Polish(    rng, smeared_normal, &direct, &normal, value, ctx ); break ; 
        }
    }

    a->save("$FOLD/q.npy"); 
}



inline void QSim_MockTest::SmearNormal_SigmaAlpha_one()
{
    rng.setSequenceIndex(0); 

    float sigma_alpha = 0.1f ; 
    float3 direct = make_float3(0.f, 0.f, -1.f ); 
    float3 normal = make_float3(0.f, 0.f,  1.f ); 

    float3 smeared_normal = make_float3( 0.f, 0.f, 0.f ) ; 
    sctx ctx ; 
    ctx.idx = 0 ; 

    sim->SmearNormal_SigmaAlpha(rng, &smeared_normal, &direct, &normal, sigma_alpha, ctx );
}

inline void QSim_MockTest::run()
{
    if(     strcmp(CHECK,"smear_normal_sigma_alpha")==0) SmearNormal(0, 0.1f) ;   
    else if(strcmp(CHECK,"smear_normal_polish")==0)     SmearNormal(1, 0.8f) ; 
    else
    {
        std::cerr 
            << "QSim_MockTest::run" 
            << " CHECK " << ( CHECK ? CHECK : "-" ) 
            << " UNHANDLED " 
            << std::endl
            ;
    } 

}



int main(int argc, char** argv)
{
    QSim_MockTest t ; 
    std::cout << t.desc() ; 

    /*
    t.propagate_at_surface_CustomART_manual() ; 
    t.propagate_at_boundary_manual() ; 
    t.fill_state() ; 
    t.propagate_at_boundary() ; 
    t.propagate() ; 
    t.SmearNormal_debug() ; 
    t.SmearNormal_SigmaAlpha_one() ; 
    */

    t.run(); 

    return 0 ; 
}


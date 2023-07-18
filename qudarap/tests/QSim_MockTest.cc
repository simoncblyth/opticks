/**
QSim_MockTest.cc : CPU tests of QSim.hh/qsim.h CUDA code using MOCK_CURAND mocking 
=======================================================================================

HMM: QSim.hh not very amenable to standalone use because the boatload of 
headers that come with it : so operating at lower level.

Standalone compile and run with::

   ./QSim_MockTest.sh 

**/

#include "NPFold.h"

#include "scuda.h"
#include "smath.h"
#include "squad.h"
#include "srec.h"
#include "stag.h"
#include "sflow.h"
#include "sphoton.h"
#include "scurand.h"    // this brings in s_mock_curand.h for CPU 
#include "SPMT.h"
#include "SBnd.h"
#include "OpticksPhoton.hh"

#include "QPMT.hh"
#include "qpmt.h"
#include "qsim.h"

struct QSim_MockTest
{
    static constexpr const char* BND = 
    "Pyrex/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/HamamatsuR12860_PMT_20inch_photocathode_mirror_logsurf/Vacuum" ;

    curandStateXORWOW rng ; 
    const NP* bnd ; 
    const SBnd* sbnd ; 
    int bnd_idx ; 
    const NPFold* jpmt ; 
    const QPMT<float>* qpmt ; 
    qsim* sim ; 

    QSim_MockTest(); 
    void init(); 
    std::string desc() const; 

    void generate_photon_dummy(); 
    void uniform_sphere();
    void propagate_at_boundary();
    void propagate_at_surface_CustomART();   
};

inline QSim_MockTest::QSim_MockTest()
    :
    rng(1u),
    bnd(NP::Load("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard/bnd.npy")),
    sbnd(bnd ? new SBnd(bnd) : nullptr),
    bnd_idx(sbnd ? sbnd->getBoundaryIndex(BND) : -1),
    jpmt(SPMT::Serialize()),
    qpmt( jpmt ? new QPMT<float>( jpmt ) : nullptr),  
    sim(new qsim)
{
    init(); 
}



inline void QSim_MockTest::init()
{
    assert( bnd ); 
    assert( bnd_idx > -1 ); 
    assert( qpmt ); 

    sim->pmt = qpmt->d_pmt ; 
    rng.set_fake(0.); // 0/1:forces transmit/reflect 
}

inline std::string QSim_MockTest::desc() const
{
    std::stringstream ss ; 
    ss << "QSim_MockTest::desc" << std::endl 
       << " bnd_idx " << bnd_idx << std::endl  
       << " sbnd.getBoundarySpec " << ( sbnd ? sbnd->getBoundarySpec(bnd_idx) : "-" )
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
QSim_MockTest::propagate_at_boundary
----------------------------------------

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
inline void QSim_MockTest::propagate_at_boundary()
{
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 
    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)) ; 

    std::cout 
        << " QSim_MockTest::propagate_at_boundary "
        << " nrm " << nrm 
        << " mom " << mom 
        << std::endl 
        ;

    quad2 prd ; 
    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;  

    sctx ctx ; 
    ctx.prd = &prd ; 

    sstate& s = ctx.s ; 
    s.material1.x = 1.0f ; 
    s.material2.x = 1.5f ; 
    // The two RINDEX are the only state needed for propagate_at_boundary  
    // TODO: get these in mocked up way from the bnd texture 


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

inline void QSim_MockTest::propagate_at_surface_CustomART()
{
    float3 nrm = make_float3(0.f, 0.f, 1.f ); // surface normal in +z direction 
    float distance = 0.1f ; 
    float lposcost = 0.5f ; 
    int identity = 1001 ; 
    int boundary = bnd_idx ; 

    float3 pos = make_float3( 0.f, 0.f, 0.f ); 
    float3 mom = normalize(make_float3(1.f, 0.f, -1.f)); 
    float3 pol = normalize(make_float3(0.f, 1.f,  0.f)); 
    const float wavelength_nm = 440.f ; 

    // lpmtcat doesnt matter as Pyrex and Vacuum are same for all of them 
    float n0 = sim->pmt->get_lpmtcat_rindex_wl( 0, 0, 0, wavelength_nm ); 
    float n3 = sim->pmt->get_lpmtcat_rindex_wl( 0, 3, 0, wavelength_nm ); 

    std::cout 
        << "QSim_MockTest::propagate_at_surface_CustomART "
        << " boundary " << boundary 
        << " nrm " << nrm 
        << " mom " << mom 
        << " n0 " << std::fixed << std::setw(10) << std::setprecision(4) << n0 
        << " n3 " << std::fixed << std::setw(10) << std::setprecision(4) << n3 
        << std::endl 
        ;

    quad2 prd ; 
    prd.q0.f.x = nrm.x ; 
    prd.q0.f.y = nrm.y ; 
    prd.q0.f.z = nrm.z ;  
    prd.q0.f.w = distance ; 

    prd.q1.f.x = lposcost ; 
    prd.q1.u.y = 0u ; 
    prd.q1.u.z = identity ; 
    prd.q1.u.w = boundary ; 

    sctx ctx ; 
    ctx.prd = &prd ; 
    ctx.s.material1.x = n0 ;  // Pyrex RINDEX
    ctx.s.material2.x = n3 ;  // Vacuum RINDEX 

    ctx.p.zero(); 
    ctx.p.pos = pos ; 
    ctx.p.mom = mom ; 
    ctx.p.pol = pol ; 
    ctx.p.wavelength = wavelength_nm ; 

    unsigned flag = 0 ;  
    int ctrl = sim->propagate_at_surface_CustomART(flag, rng, ctx) ; 
 
    std::cout 
        << "QSim_MockTest::propagate_at_surface_CustomART "
        << " flag " << flag << " : " << OpticksPhoton::Flag(flag) 
        << " ctrl " << ctrl << " : " << sflow::desc(ctrl)  
        << std::endl 
        ;
}

int main(int argc, char** argv)
{
    QSim_MockTest t ; 
    std::cout << t.desc() ; 
    //t.propagate_at_surface_CustomART() ; 
    t.propagate_at_boundary() ; 

    return 0 ; 
}


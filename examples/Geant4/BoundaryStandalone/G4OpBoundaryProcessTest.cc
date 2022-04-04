/**
G4OpBoundaryProcessTest.cc
============================

Attempt to minimally mockup Geant4 environment needed for G4OpBoundaryProcess::PostStepDoIt 
looks too difficult because of the access to the surface normal G4Navigator::GetGlobalExitNormal
plus also there is optical surface checking too. 

Probably easiest to setup a "proper" Geant4 geometry to test within. 

**/

#include <cstring>
#include <iostream>
#include <iomanip>

#include "G4OpBoundaryProcess_MOCK.hh"
#include "X4OpBoundaryProcessStatus.hh"

#include "QSimLaunch.hh"
#include "X4OpticalSurface.hh"
#include "X4OpticalSurfaceModel.hh"
#include "X4OpticalSurfaceFinish.hh"
#include "X4SurfaceType.hh"


#include "G4OpticalPhoton.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"
#include "G4TouchableHandle.hh"

#include "G4RandomTools.hh"       // G4LambertianRand
#include "G4RandomDirection.hh"

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"
#include "OpticksUtil.hh"
#include "OpticksRandom.hh"
#include "NP.hh"

#include "vector_functions.h"
#include "scuda.h"
#include "squad.h"


struct G4OpBoundaryProcessTest 
{
    static unsigned Status(unsigned status); 

    static NP*      MakeValueArray(double value); 

    static G4MaterialPropertiesTable* MakeOpticalMPT(
        const G4MaterialPropertyVector* v_reflectivity,    
        const G4MaterialPropertyVector* v_efficiency,    
        const G4MaterialPropertyVector* v_transmittance
       ); 
    static G4OpticalSurface* MakeOpticalSurface(G4MaterialPropertiesTable* mpt) ;
    void setOpticalSurfaceFinish(const char* finish_); 
    unsigned getOpticalSurfaceFinish() const ; 

    const char*       name ; 
    unsigned          test ;  
    bool              surftest ; 
    int               num ; 
    OpticksRandom*    rnd ; 
    const char*       srcdir ; 
    const char*       dstdir ; 
    float3            normal ; 

    // Materials setup 
    float             n1 ; 
    float             n2 ; 
    const NP*         a_rindex1 ; 
    const NP*         a_rindex2 ; 
    const G4MaterialPropertyVector* rindex1 ; 
    const G4MaterialPropertyVector* rindex2 ;
    const G4Material* material1 ; 
    const G4Material* material2 ; 
    G4Material*       material1_ ; 
    G4Material*       material2_ ; 


    // OpticalSurface setup
    float             reflectivity ;
    float             efficiency ; 
    float             transmittance ; 

    int               eload ; 

    const NP*         a_reflectivity ; 
    const NP*         a_efficiency ; 
    const NP*         a_transmittance ; 

    const G4MaterialPropertyVector* v_reflectivity ; 
    const G4MaterialPropertyVector* v_efficiency ; 
    const G4MaterialPropertyVector* v_transmittance ; 

    G4MaterialPropertiesTable*      optical_mpt ; 
    G4OpticalSurface*               optical_surf ; 

    // process
    G4OpBoundaryProcess_MOCK* proc ; 

    // ephoton
    quad4 p0 ; 


    std::string desc() const ; 
    G4OpBoundaryProcessTest(const char* srcdir_ekey, const char* dstdir_ekey); 
    void init(); 
    void init_normal(); 
    void init_prd_normal(); 
    void init_surface();
 
    void load( std::vector<quad4>& pp, const char* npy_name ); 

    void propagate_at_boundary_(quad4& photon, int idx);

    void dump( const quad4& photon, int idx ); 
    void save(const quad4& p, const char* name) const ; 
    void save_photon( const float* data, unsigned num_photon ); 

    void photon_mutate_(quad4& p, int idx); 
    int  photon_mutate(int idx); 

    void quad_generate(quad& q, unsigned idx); 
    int  quad_generate(); 

    int run(); 
}; 

/**
G4OpBoundaryProcessTest::Status
--------------------------------

Note info loss here, should probably store origin status for debug purposes too. 

**/
unsigned G4OpBoundaryProcessTest::Status(unsigned status)
{
    unsigned st = 0 ; 
    switch(status)
    {
        case FresnelReflection:       
        case TotalInternalReflection: 
        case LambertianReflection:   
        case SpikeReflection: 
                                      st = BOUNDARY_REFLECT  ; break ;
        case FresnelRefraction:       
        case Transmission:            
                                      st = BOUNDARY_TRANSMIT ; break ; 

        case Absorption:              st = SURFACE_ABSORB    ; break ; 
        case Detection:               st = SURFACE_DETECT    ; break ; 
    }
    return st ; 
}

NP* G4OpBoundaryProcessTest::MakeValueArray(double value)  // static
{
    NP* a = NP::Make<double>( 1.55/1e6, value, 15.5/1e6, value ) ; 
    a->change_shape(-1, 2); 
    a->pdump("G4OpBoundaryProcessTest::MakeValueArray" , 1e6 ); 
    return a ; 
}

/**
HMM the Opticks and Geant4 approach are a bit divergent wrt specular and diffuse reflectivity 
so to match have to carefully arrange prop values in coordination with the surface finish.  

Relevant Geant4 property enum::     

   kREFLECTIVITY  (or kREALRINDEX, kIMAGINARYRINDEX)
   kEFFICIENCY
   kTRANSMITTANCE
**/

G4MaterialPropertiesTable* G4OpBoundaryProcessTest::MakeOpticalMPT(
    const G4MaterialPropertyVector* v_reflectivity,    
    const G4MaterialPropertyVector* v_efficiency,    
    const G4MaterialPropertyVector* v_transmittance
) 
{
    return OpticksUtil::MakeMaterialPropertiesTable( "REFLECTIVITY", v_reflectivity, "EFFICIENCY", v_efficiency, "TRANSMITTANCE", v_transmittance ); 
}

G4OpticalSurface* G4OpBoundaryProcessTest::MakeOpticalSurface(G4MaterialPropertiesTable* mpt) // static
{
    const char* optical_surface_ = U::GetEnv("OPTICAL_SURFACE", "esurfname,glisur,polished,dielectric_dielectric,1.0" ); 

    X4OpticalSurface* xsurf = X4OpticalSurface::FromString(optical_surface_ ); 
    G4String name = xsurf->name ; 
    G4OpticalSurfaceModel model = (G4OpticalSurfaceModel)X4OpticalSurfaceModel::Model(xsurf->model) ; 
    G4OpticalSurfaceFinish finish = (G4OpticalSurfaceFinish)X4OpticalSurfaceFinish::Finish(xsurf->finish); 
    G4SurfaceType type = (G4SurfaceType)X4SurfaceType::Type(xsurf->type) ; 
    G4double value = std::atof(xsurf->value) ; 

    G4OpticalSurface* os = new G4OpticalSurface(name, model, finish, type, value); 
    os->SetMaterialPropertiesTable(mpt); 

    return os ; 
}

void G4OpBoundaryProcessTest::setOpticalSurfaceFinish(const char* finish_)
{
    G4OpticalSurfaceFinish finish = (G4OpticalSurfaceFinish)X4OpticalSurfaceFinish::Finish(finish_) ; 
    std::cout << "G4OpBoundaryProcessTest::setOpticalSurfaceFinish " << finish_ << std::endl ; 
    optical_surf->SetFinish(finish); 
}

unsigned G4OpBoundaryProcessTest::getOpticalSurfaceFinish() const 
{
    assert(optical_surf); 
    return optical_surf->GetFinish() ; 
}



std::string G4OpBoundaryProcessTest::desc() const 
{
    std::stringstream ss ; 
    ss 
        << " name "   << ( name   ? name : "-" ) 
        << " test " << test 
        << " surftest " << surftest 
        << " srcdir " << ( srcdir ? srcdir : "-" ) 
        << " dstdir " << ( dstdir ? dstdir : "-" )
        << " normal ("
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << normal.x 
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << normal.y
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << normal.z
        << ")" 
        << " n1 " << std::setw(10) << std::fixed << std::setprecision(4) << n1
        << " n2 " << std::setw(10) << std::fixed << std::setprecision(4) << n2
        ; 

    std::string s = ss.str(); 
    return s ; 
}

G4OpBoundaryProcessTest::G4OpBoundaryProcessTest(const char* srcdir_ekey, const char* dstdir_ekey)
    :
    name(getenv("TEST")),
    test(QSimLaunch::Type(name)),
    surftest(QSimLaunch::IsSurface(test)),
    num(qenvint("NUM", "8")),
    rnd(OpticksRandom::Enabled() ? new OpticksRandom : nullptr ),
    srcdir(getenv(srcdir_ekey)),
    dstdir(getenv(dstdir_ekey)),
    normal(make_float3(0.f, 0.f, 1.f)),   // may be overrden by normal from init_prd_normal
    n1(qenvfloat("M1_REFRACTIVE_INDEX","1.0")),
    n2(qenvfloat("M2_REFRACTIVE_INDEX","1.5")),
    a_rindex1(MakeValueArray(n1)), 
    a_rindex2(MakeValueArray(n2)), 
    rindex1(OpticksUtil::MakeProperty(a_rindex1)),
    rindex2(OpticksUtil::MakeProperty(a_rindex2)),
    material1(OpticksUtil::MakeMaterial(rindex1, "Material1")),
    material2(OpticksUtil::MakeMaterial(rindex2, "Material2")),
    material1_(const_cast<G4Material*>(material1)),
    material2_(const_cast<G4Material*>(material2)),

    reflectivity(0.f),
    efficiency(0.f),
    transmittance(0.f),

    eload(qvals3(reflectivity,efficiency,transmittance, "REFLECTIVITY_EFFICIENCY_TRANSMITTANCE", "1,0,0")),

    a_reflectivity(MakeValueArray(reflectivity)),
    a_efficiency(MakeValueArray(efficiency)),
    a_transmittance(MakeValueArray(transmittance)),

    v_reflectivity(OpticksUtil::MakeProperty(a_reflectivity)),
    v_efficiency(OpticksUtil::MakeProperty(a_efficiency)),
    v_transmittance(OpticksUtil::MakeProperty(a_transmittance)),
  
    optical_mpt(MakeOpticalMPT(v_reflectivity,v_efficiency,v_transmittance)),
    optical_surf(MakeOpticalSurface(optical_mpt)), 

    proc(new G4OpBoundaryProcess_MOCK()),
    p0(quad4::make_ephoton())              // load initial photon p0 from envvars 
{
    init(); 
}
void G4OpBoundaryProcessTest::init()
{
    init_normal(); 
    init_surface(); 
    std::cout 
        << "G4OpBoundaryProcessTest::init "  << desc() 
        << std::endl 
        ;
}

void G4OpBoundaryProcessTest::init_normal()
{
    if( srcdir != nullptr )
    {
        std::cout << "G4OpBoundaryProcessTest::init booting from srcdir " << srcdir << std::endl ; 
        init_prd_normal(); 
    } 
    else
    {
        qvals( normal, "NRM" , "0,0,1") ; 
        std::cout << "G4OpBoundaryProcessTest::init no srcdir : use normal from NRM evar " << std::endl ; 
    }

    //rnd->m_flat_debug = true  ;   // when true dumps a line for every G4UniformRand call 
    proc->theGlobalNormal_MOCK.set( normal.x, normal.y, normal.z ); 
}

void G4OpBoundaryProcessTest::init_prd_normal()
{
    std::vector<quad4> prds ;
    load(prds, "prd.npy"); 
    assert( prds.size() == 1 );
 
    const quad4& prd = prds[0]; 
    const float3* nrm = (float3*)&prd.q0.f.x ;
 
    normal.x = nrm->x ; 
    normal.y = nrm->y ; 
    normal.z = nrm->z ; 

    NP::Write(dstdir, "prd.npy",  (float*)&prd.q0.f.x, 1, 4, 4  ); // save the prd to dstfold for python consumption
}


void G4OpBoundaryProcessTest::init_surface()
{
    proc->OpticalSurface_MOCK = surftest ? optical_surf  : nullptr ;   // G4OpticalSurface
    if( test == REFLECT_DIFFUSE )
    {
        assert( getOpticalSurfaceFinish() == groundfrontpainted ) ; 
    }
    else if( test == REFLECT_SPECULAR )
    {
        assert( getOpticalSurfaceFinish() == polishedfrontpainted ) ; 
    }
}





void G4OpBoundaryProcessTest::load( std::vector<quad4>& pp, const char* npy_name )
{
    if( srcdir == nullptr )
    {
        pp.resize(num); 
        for(unsigned i=0 ; i < num ; i++ ) pp[i] = p0 ; 
    }
    else
    { 
        NP* a = NP::Load(srcdir, npy_name) ; 
        assert( a->has_shape(-1,4,4) ); 
        unsigned ni = a->shape[0] ; 
        pp.resize(ni); 
        float* pp_data = (float*)pp.data() ;
        a->write<float>(pp_data); 
    }
    std::cout << "G4OpBoundaryProcessTest::load " << pp.size() << std::endl ;  
}
 
/**
G4OpBoundaryProcessTest::propagate_at_boundary_
-------------------------------------------------

Just leaking as Geant4 not keen on being mocked like this with objects on stack.

**/

void G4OpBoundaryProcessTest::propagate_at_boundary_(quad4& photon, int idx)
{
    float3* mom0 = (float3*)&photon.q0.f.x ; 
    float3* mom = (float3*)&photon.q1.f.x ; 
    float3* pol = (float3*)&photon.q2.f.x ; 
    float3* pol0 = (float3*)&photon.q3.f.x ; 

    // take copy of the initial mom and pol 
    mom0->x = photon.q1.f.x ; 
    mom0->y = photon.q1.f.y ; 
    mom0->z = photon.q1.f.z ; 

    pol0->x = photon.q2.f.x ; 
    pol0->y = photon.q2.f.y ; 
    pol0->z = photon.q2.f.z ; 

    G4double en = 1.*MeV ; 
    G4ParticleMomentum momentum(mom->x,mom->y,mom->z); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4OpticalPhoton::Definition(),momentum);
    particle->SetPolarization(pol->x, pol->y, pol->z );  
    particle->SetKineticEnergy(en); 

    //G4ThreeVector position(pos->x, pos->y, pos->z); 
    G4ThreeVector position(0.f, 0.f, 0.f); 
    G4double time(0.); 

    G4Track* track = new G4Track(particle,time,position);
    G4StepPoint* pre = new G4StepPoint ; 
    G4StepPoint* post = new G4StepPoint ; 

    G4ThreeVector pre_position(0., 0., 0.);
    G4ThreeVector post_position(0., 0., 1.);
    pre->SetPosition(pre_position); 
    post->SetPosition(post_position); 

    G4VPhysicalVolume* prePV = nullptr ; 
    G4VPhysicalVolume* postPV = nullptr ; 

    G4NavigationHistory* pre_navHist = new G4NavigationHistory ; 
    pre_navHist->SetFirstEntry( prePV ); 

    G4NavigationHistory* post_navHist = new G4NavigationHistory ;  
    post_navHist->SetFirstEntry( postPV ); 

    G4TouchableHistory* pre_touchHist = new G4TouchableHistory(*pre_navHist);  
    G4TouchableHistory* post_touchHist = new G4TouchableHistory(*post_navHist);  

    G4TouchableHandle pre_touchable(pre_touchHist);  
    G4TouchableHandle post_touchable(post_touchHist);  

    pre->SetTouchableHandle(pre_touchable); 
    post->SetTouchableHandle(post_touchable); 

    const G4StepStatus postStepStatus = fGeomBoundary ; 
    post->SetStepStatus( postStepStatus ); 

    // G4Track::GetMaterial comes from current step preStepPoint 
    pre->SetMaterial(material1_);    // HUH: why not const 
    post->SetMaterial(material2_); 

    G4double step_length = 1. ; 

    G4Step* step = new G4Step ; 
    step->SetPreStepPoint(pre);
    step->SetPostStepPoint(post);
    step->SetStepLength(step_length);
    track->SetStepLength(step_length); 
    track->SetStep(step);

    G4VParticleChange* change = proc->PostStepDoIt(*track, *step) ;

    G4OpBoundaryProcess_MOCKStatus theStatus = proc->GetStatus(); 
    unsigned theFlag = Status(theStatus) ; 

    if( theFlag == 0)
    {
       bool tir = theStatus == TotalInternalReflection ; 
        std::cout 
             << "G4OpBoundaryProcessTest::propagate_at_boundary_"
             << " theFlag is ZERO " 
             << " theStatus " << theStatus  
             << " theStatus.Name "  << X4OpBoundaryProcessStatus::Name( theStatus ) 
             << " theFlag " << theFlag 
             << " tir " << tir 
             << std::endl 
             ;  
    }

    G4ParticleChange* pc = dynamic_cast<G4ParticleChange*>(change);  
    const G4ThreeVector* smom = pc->GetMomentumDirection();
    const G4ThreeVector* spol = pc->GetPolarization();

    mom->x =  smom->x() ; 
    mom->y =  smom->y() ; 
    mom->z =  smom->z() ; 

    pol->x =  spol->x() ; 
    pol->y =  spol->y() ; 
    pol->z =  spol->z() ; 

    photon.q3.u.w = theFlag ; 
    photon.q1.f.w = proc->theTransCoeff_MOCK ; 

}

void G4OpBoundaryProcessTest::dump( const quad4& photon, int idx )
{
    unsigned status = photon.q3.u.w ; 
    float flat_prior = photon.q0.f.w ; 

    //float3* pos = (float3*)&photon.q0.f.x ; 
    float3* mom = (float3*)&photon.q1.f.x ; 
    float3* pol = (float3*)&photon.q2.f.x ; 

    std::cout 
        << " i " << std::setw(6) << idx 
        << " s " << std::setw(2) << status 
        << " " << std::setw(16) << OpticksPhoton::Flag( status ) 
        << " " << std::setw(10) << std::setprecision(3) << flat_prior
        << " mom " 
        << " " << std::setw(10) << std::setprecision(3) << mom->x
        << " " << std::setw(10) << std::setprecision(3) << mom->y
        << " " << std::setw(10) << std::setprecision(3) << mom->z
        << " pol " 
        << " " << std::setw(10) << std::setprecision(3) << pol->x
        << " " << std::setw(10) << std::setprecision(3) << pol->y
        << " " << std::setw(10) << std::setprecision(3) << pol->z
        << std::endl 
        ;
}

void G4OpBoundaryProcessTest::save(const quad4& p, const char* name) const 
{
    assert(dstdir); 
    NP::Write( dstdir, name,  (float*)&p.q0.f.x , 1, 4, 4  );
}

void G4OpBoundaryProcessTest::save_photon( const float* data, unsigned num_photon )
{
    assert(dstdir); 
    NP* p = NP::Make<float>( num_photon, 4, 4); 
    p->read(data); 
    p->set_meta<float>("normal_x", normal.x ); 
    p->set_meta<float>("normal_y", normal.y ); 
    p->set_meta<float>("normal_z", normal.z ); 
    p->set_meta<float>("n1", n1 ); 
    p->set_meta<float>("n2", n2 ); 
    p->save( dstdir, "p.npy" ); 
}


/**
G4OpBoundaryProcessTest::photon_mutate_
-----------------------------------------

Unlike with quadrap/QSim.cc do not have convenient API methods to fine grain test different things, 
as most everything happens via G4OpBoundaryProcess::PostStepDoIt 
Thus instead need to setup the Geant4 environment in such a way to induce the desired thing to happen. 

**/

void G4OpBoundaryProcessTest::photon_mutate_(quad4& p, int idx)
{
    assert( idx > -1 ); 
    proc->photon_idx = idx ; 
    if(rnd) rnd->setSequenceIndex(idx);  // arranges use of pre-cooked randoms by G4UniformRand (hijacks the engine)

    switch(test)
    {
        case PROPAGATE_AT_BOUNDARY_S_POLARIZED:
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:
        case PROPAGATE_AT_BOUNDARY_X_POLARIZED:
        case REFLECT_DIFFUSE:
        case REFLECT_SPECULAR:
                                               propagate_at_boundary_(p, idx);    break ; 
        default:
                std::cout << "G4OpBoundaryProcessTest::photon_mutate_ ERR not handled test " << test << " name " << name << std::endl ;  
    }

    double flat_prior = rnd ? rnd->getFlatPrior() : -1. ; 
    p.q0.f.w = flat_prior ; 
  
    if(rnd) rnd->setSequenceIndex(-1);  // disable random hijacking 

    bool dump_ = idx < 10 || ( idx % 10000 == 0 ) ;  
    if(dump_) dump(p, idx); 
}

/**
G4OpBoundaryProcessTest::photon_mutate
----------------------------------------

Initial photons are loaded from file when srcdir 
is defined or otherwise duplicated from the p0 ephoton.

For default idx of -1 all photons are mutated, otherwise
only the idx photon is mutated. 

**/

int G4OpBoundaryProcessTest::photon_mutate(int idx)
{
    std::vector<quad4> pp ;
    load(pp, "p.npy");    
    unsigned num_photon = pp.size()  ; 
    if( idx == -1 )
    {
        for(unsigned i=0 ; i < num_photon ; i++)
        {
            quad4& p = pp[i] ; 
            photon_mutate_(p, i); 
        }
        float* pp_data = (float*)pp.data() ;
        save_photon( pp_data, num_photon ); 
    }
    else
    {
        quad4& p = pp[idx] ; 
        photon_mutate_(p, idx); 
    }
    return 0 ; 
}

void G4OpBoundaryProcessTest::quad_generate(quad& q, unsigned idx)
{
    if(rnd) rnd->setSequenceIndex(idx);  // arranges use of pre-cooked randoms by G4UniformRand (hijacks the engine)

    G4ThreeVector dir ; 
    if( test == RANDOM_DIRECTION_MARSAGLIA )
    {
        dir = G4RandomDirection() ;   
    }
    else if( test == LAMBERTIAN_DIRECTION )
    {
        const G4ThreeVector& normal =  proc->theGlobalNormal_MOCK ; 
        dir = G4LambertianRand(normal); 
    }
    q.f.x = dir.x(); 
    q.f.y = dir.y(); 
    q.f.z = dir.z(); 
    q.u.w = idx ; 


    if(rnd) rnd->setSequenceIndex(-1);
}

int G4OpBoundaryProcessTest::quad_generate()
{
    NP* q = NP::Make<float>( num, 4 ); 
    quad* qq = (quad*)q->values<float>(); 
    for(unsigned idx=0 ; idx < num ; idx++) 
    {
        quad_generate(qq[idx], idx) ; 
    }
    assert(dstdir); 
    q->save( dstdir, "q.npy" ); 
    return 0 ; 
}

int G4OpBoundaryProcessTest::run()
{
    std::cout << "desc " << desc() << std::endl ; 
    int rc = 0 ;     
    switch(test)
    {
        case PROPAGATE_AT_BOUNDARY_S_POLARIZED:
        case PROPAGATE_AT_BOUNDARY_P_POLARIZED:
        case PROPAGATE_AT_BOUNDARY_X_POLARIZED:
        case REFLECT_DIFFUSE:
        case REFLECT_SPECULAR:
                                         rc = photon_mutate(-1) ; break ; 
        case RANDOM_DIRECTION_MARSAGLIA: 
        case LAMBERTIAN_DIRECTION:       
                                         rc = quad_generate()  ; break ; 
        default:
                                         rc = 666 ;  
    }
    std::cout << "desc " << desc() << std::endl ; 
    if( rc != 0 )
    {
        std::cout 
            << "G4OpBoundaryProcessTest::run ERROR"
            << " test " << test 
            << " name " << name 
            << " rc " << rc 
            << std::endl 
            ; 
    }

    return rc ; 
}


int main(int argc, char** argv)
{
    G4OpBoundaryProcessTest t("OPTICKS_BST_SRCDIR", "OPTICKS_BST_DSTDIR") ; 
    return t.run(); 
}

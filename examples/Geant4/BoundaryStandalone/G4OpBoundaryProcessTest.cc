/**
G4OpBoundaryProcessTest.cc
============================

Attempt to minimally mockup Geant4 environment needed for G4OpBoundaryProcess::PostStepDoIt 
looks too difficult because of the access to the surface normal G4Navigator::GetGlobalExitNormal
plus also there is optical surface checking too. 

Probably easiest to setup a "proper" Geant4 geometry to test within. 

**/
#include <iostream>
#include <iomanip>

#include "G4OpBoundaryProcess_MOCK.hh"
#include "X4OpBoundaryProcessStatus.hh"

#include "G4OpticalPhoton.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"
#include "G4NavigationHistory.hh"
#include "G4TouchableHistory.hh"
#include "G4TouchableHandle.hh"

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
    static NP*      MakeRindexArray(double rindex); 

    const char*       srcdir ; 
    const char*       dstdir ; 
    float3            normal ; 
    OpticksRandom*    rnd ; 
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
    G4OpBoundaryProcess_MOCK* proc ; 


    std::string desc() const ; 
    G4OpBoundaryProcessTest(const char* srcdir_ekey, const char* dstdir_ekey); 
    void init(); 
    void init_prd_normal(); 
    void load( std::vector<quad4>& pp, const char* npy_name ); 

    void propagate_at_boundary( quad4& photon, int idx); 
    void propagate_at_boundary_(quad4& photon, int idx);
    void propagate_at_boundary(int idx); 
    void propagate_at_boundary_ephoton(); 

    void dump( const quad4& photon, int idx ); 
    void save(const quad4& p, const char* name) const ; 
    void save_photon( const float* data, unsigned num_photon ); 
}; 


unsigned G4OpBoundaryProcessTest::Status(unsigned status)
{
    unsigned st = 0 ; 
    switch(status)
    {
        case FresnelReflection:       st = BOUNDARY_REFLECT  ; break ; 
        case TotalInternalReflection: st = BOUNDARY_REFLECT  ; break ; 
        case FresnelRefraction:       st = BOUNDARY_TRANSMIT ; break ; 
    }
    return st ; 
}

NP* G4OpBoundaryProcessTest::MakeRindexArray(double rindex)  // static
{
    NP* a = NP::Make<double>( 1.55/1e6, rindex, 15.5/1e6, rindex ) ; 
    a->change_shape(-1, 2); 
    a->pdump("G4OpBoundaryProcessTest::MakeRindexArray" , 1e6 ); 
    return a ; 
}

std::string G4OpBoundaryProcessTest::desc() const 
{
    std::stringstream ss ; 
    ss 
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
    srcdir(getenv(srcdir_ekey)),
    dstdir(getenv(dstdir_ekey)),
    normal(make_float3(0.f, 0.f, 1.f)),   // may be overrden by normal from init_prd_normal
    rnd(new OpticksRandom),
    n1(qenvfloat("M1_REFRACTIVE_INDEX","1.0")),
    n2(qenvfloat("M2_REFRACTIVE_INDEX","1.5")),
    a_rindex1(MakeRindexArray(n1)), 
    a_rindex2(MakeRindexArray(n2)), 
    rindex1(OpticksUtil::MakeProperty(a_rindex1)),
    rindex2(OpticksUtil::MakeProperty(a_rindex2)),
    material1(OpticksUtil::MakeMaterial(rindex1, "Material1")),
    material2(OpticksUtil::MakeMaterial(rindex2, "Material2")),
    material1_(const_cast<G4Material*>(material1)),
    material2_(const_cast<G4Material*>(material2)),
    proc(new G4OpBoundaryProcess_MOCK())
{
    init(); 
}


void G4OpBoundaryProcessTest::init()
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

    std::cout 
        << "G4OpBoundaryProcessTest::init "  << desc() 
        << std::endl 
        ;
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

void G4OpBoundaryProcessTest::load( std::vector<quad4>& pp, const char* npy_name )
{
    assert( srcdir ); 
    NP* a = NP::Load(srcdir, npy_name) ; 
    assert( a->has_shape(-1,4,4) ); 
    unsigned ni = a->shape[0] ; 
    pp.resize(ni); 
    float* pp_data = (float*)pp.data() ;
    a->write<float>(pp_data); 
}
 
void G4OpBoundaryProcessTest::propagate_at_boundary(quad4& photon, int idx)
{
    assert( idx > -1 ); 
    proc->photon_idx = idx ; 
    rnd->setSequenceIndex(idx);  // arranges use of pre-cooked randoms by G4UniformRand (hijacks the engine)

    propagate_at_boundary_( photon, idx ); 

    double flat_prior = rnd->getFlatPrior(); 
    photon.q0.f.w = flat_prior ; 
    rnd->setSequenceIndex(-1);  // disable random hijacking 

    bool dump_ = idx < 10 || ( idx % 10000 == 0 ) ;  
    if(dump_) dump(photon, idx); 
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
             << " theStatus "  << X4OpBoundaryProcessStatus::Name( theStatus ) 
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

void G4OpBoundaryProcessTest::propagate_at_boundary(int idx)
{
    std::vector<quad4> pp ;
    load(pp, "p.npy"); 
    unsigned num_photon = pp.size(); 

    if( idx == -1 )
    {
        for(unsigned i=0 ; i < num_photon ; i++)
        {
            quad4& p = pp[i] ; 
            propagate_at_boundary(p, i);   // mutates the photons
        }
        float* pp_data = (float*)pp.data() ;
        save_photon( pp_data, num_photon ); 

    }
    else
    {
        // just the idx photon for debugging 
        quad4& p = pp[idx] ; 
        propagate_at_boundary(p, idx);   // mutates the photons
    }
}

void G4OpBoundaryProcessTest::propagate_at_boundary_ephoton()
{
    int num_photon = qenvint("NUM", "8");
    std::cout << "G4OpBoundaryProcessTest::propagate_at_boundary_repeat NUM " << num_photon << std::endl ; 

    quad4 p0 ; 
    p0.ephoton(); // load initial photon p0 from envvars 

    std::vector<quad4> pp(num_photon) ;
    for(unsigned idx=0 ; idx < num_photon ; idx++)
    {
        pp[idx] = p0 ; 
        quad4& p = pp[idx] ; 
        propagate_at_boundary(p, idx);   // mutates the photons
    } 

    float* pp_data = (float*)pp.data() ;
    save_photon( pp_data, num_photon ); 
}

int main(int argc, char** argv)
{
    G4OpBoundaryProcessTest t("OPTICKS_BST_SRCDIR", "OPTICKS_BST_DSTDIR") ; 
    std::cout << "t.desc " << t.desc() << std::endl ; 

    t.proc->photon_idx_debug = -1 ;  // index for debug output  

    if(t.srcdir == nullptr )
    {
        t.propagate_at_boundary_ephoton();   
    }
    else
    {
        t.propagate_at_boundary(-1); 
      //t.propagate_at_boundary(251959); 
    }

    std::cout << "t.desc " << t.desc() << std::endl ; 
    return 0 ; 
}

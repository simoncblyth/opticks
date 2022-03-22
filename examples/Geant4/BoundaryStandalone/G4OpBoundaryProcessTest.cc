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

#include "OpticksUtil.hh"
#include "OpticksRandom.hh"
#include "NP.hh"


struct float3 { float x,y,z ;  };
struct float4 { float x,y,z,w ;  };
struct uint4 {  unsigned x,y,z,w ;  };
float3 make_float3(float x, float y, float z)          { float3 v ; v.x = x ; v.y = y ; v.z = z ;           return v ; } 
float4 make_float4(float x, float y, float z, float w ){ float4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
uint4  make_uint4(unsigned x, unsigned y, unsigned z, unsigned w ){ uint4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
union quad { float4 f ; uint4  u ;  }; 
struct quad4 { quad q0, q1, q2, q3 ; }; 


struct G4OpBoundaryProcessTest 
{
    static const char* FOLD ; 
    static NP* MakeRindexArray(double rindex); 

    const float3&   pos ; 
    const float3&   mom ; 
    const float3&   pol ; 
    const float3&   nrm ; 

    OpticksRandom*  rnd ; 

    const NP* a_rindex1 ; 
    const NP* a_rindex2 ; 
    const G4MaterialPropertyVector* rindex1 ; 
    const G4MaterialPropertyVector* rindex2 ;
    const G4Material* material1 ; 
    const G4Material* material2 ; 

    G4OpBoundaryProcess_MOCK* proc ; 
    G4Track*                  track ; 
    G4Step*                   step ;  

    G4OpBoundaryProcessTest(const float3& pos, const float3& mom, const float3& pol, const float3& nrm ); 

    void propagate_at_boundary(unsigned num); 
    void save_p0() const ; 

}; 


NP* G4OpBoundaryProcessTest::MakeRindexArray(double rindex)  // static
{
    NP* a = NP::Make<double>( 1.55/1e6, rindex, 15.5/1e6, rindex ) ; 
    a->change_shape(-1, 2); 
    a->pdump("G4OpBoundaryProcessTest::MakeRindexArray" , 1e6 ); 
    return a ; 
}

G4OpBoundaryProcessTest::G4OpBoundaryProcessTest(const float3& pos_, const float3& mom_, const float3& pol_, const float3& nrm_ )
    :
    pos(pos_),
    mom(mom_),
    pol(pol_),
    nrm(nrm_),
    rnd(new OpticksRandom),
    a_rindex1(MakeRindexArray(1.f)), 
    a_rindex2(MakeRindexArray(1.5f)), 
    rindex1(OpticksUtil::MakeProperty(a_rindex1)),
    rindex2(OpticksUtil::MakeProperty(a_rindex2)),
    material1(OpticksUtil::MakeMaterial(rindex1, "Material1")),
    material2(OpticksUtil::MakeMaterial(rindex2, "Material2")),
    proc(new G4OpBoundaryProcess_MOCK()),
    track(nullptr),
    step(new G4Step)
{
    //rnd->m_flat_debug = true  ;   // when true dumps a line for every G4UniformRand call 
    proc->theGlobalNormal_MOCK.set( nrm.x, nrm.y, nrm.z ); 

    G4double en = 1.*MeV ; 
    G4ParticleMomentum momentum(mom.x,mom.y,mom.z); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4OpticalPhoton::Definition(),momentum);
    particle->SetPolarization(pol.x, pol.y, pol.z );  
    particle->SetKineticEnergy(en); 

    G4ThreeVector position(pos.x, pos.y, pos.z); 
    G4double time(0.); 

    track = new G4Track(particle,time,position);

    G4StepPoint* pre = new G4StepPoint ; 
    G4StepPoint* post = new G4StepPoint ; 

    G4ThreeVector pre_position(0., 0., 0.);
    G4ThreeVector post_position(0., 0., 1.);
    pre->SetPosition(pre_position); 
    post->SetPosition(post_position); 

    // HMM : MOCKING IS INVOLVED BECAUSE NEED TO GET A SURFACE NORMAL WITH G4Navigator::GetGlobalExitNormal
    // g4-cls G4PathFinder
    // because of this moved to _MOCK the proc

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

    G4Material* material1_ = const_cast<G4Material*>(material1); 
    G4Material* material2_ = const_cast<G4Material*>(material2); 

    pre->SetMaterial(material1_);    // HUH: why not const 
    post->SetMaterial(material2_); 

    G4double step_length = 1. ; 

    step->SetPreStepPoint(pre);
    step->SetPostStepPoint(post);
    step->SetStepLength(step_length);

    track->SetStepLength(step_length); 
 
    track->SetStep(step);
}

const char* G4OpBoundaryProcessTest::FOLD = "/tmp/G4OpBoundaryProcessTest" ; 

void G4OpBoundaryProcessTest::propagate_at_boundary(unsigned num)
{
    std::vector<quad4> pp(num) ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        rnd->setSequenceIndex(i);  // arranges use of pre-cooked randoms by G4UniformRand (hijacks the engine)
        G4VParticleChange* change = proc->PostStepDoIt(*track, *step) ;
        rnd->setSequenceIndex(-1);  // disable random hijacking 

        double flat_prior = rnd->getFlatPrior(); 

        G4OpBoundaryProcess_MOCKStatus theStatus = proc->GetStatus(); 

        G4ParticleChange* pc = dynamic_cast<G4ParticleChange*>(change);  

        const G4ThreeVector* smom = pc->GetMomentumDirection();
        const G4ThreeVector* spol = pc->GetPolarization();

        bool dump = i < 10 || ( i % 1000 == 0 ) ;  
        if(dump) 
        {
            std::cout 
                << " i " << std::setw(6) << i 
                << " s " << std::setw(2) << theStatus 
                << " " << std::setw(16) << X4OpBoundaryProcessStatus::Name( theStatus ) 
                << " " << std::setw(10) << std::setprecision(3) << flat_prior
                << " mom " 
                << " " << std::setw(10) << std::setprecision(3) << smom->x() 
                << " " << std::setw(10) << std::setprecision(3) << smom->y() 
                << " " << std::setw(10) << std::setprecision(3) << smom->z()
                << " pol " 
                << " " << std::setw(10) << std::setprecision(3) << spol->x() 
                << " " << std::setw(10) << std::setprecision(3) << spol->y() 
                << " " << std::setw(10) << std::setprecision(3) << spol->z()
                << std::endl 
                ;
        }

        quad4 p ; 
        p.q0.f = make_float4( pos.x, pos.y, pos.z, float(flat_prior) );   
        p.q1.f = make_float4( smom->x(), smom->y(), smom->z(), 0.f  ) ; 
        p.q2.f = make_float4( spol->x(), spol->y(), spol->z(), 0.f  ) ; 
        p.q3.u = make_uint4(  i, 0u, 0u, theStatus );   

        pp[i] = p ; 
    }

    NP::Write( FOLD, "p.npy",  (float*)pp.data(), pp.size(), 4, 4 ); 
    save_p0(); 
}

void G4OpBoundaryProcessTest::save_p0() const 
{
    quad4 p ; 
    p.q0.f = make_float4( pos.x, pos.y, pos.z, 0.f );   
    p.q1.f = make_float4( mom.x, mom.y, mom.z, 0.f  ) ; 
    p.q2.f = make_float4( pol.x, pol.y, pol.z, 0.f  ) ; 
    p.q3.u = make_uint4(  0u, 0u, 0u, 0u );   
    NP::Write( FOLD, "p0.npy",  (float*)&p.q0.f.x , 1, 4, 4  );
}

int main(int argc, char** argv)
{
    float3 pos = make_float3( 0.f, 0.f, 0.f ); 
    float3 mom = make_float3( 1.f, 0.f, 0.f ); 
    float3 pol = make_float3( 0.f, 1.f, 0.f ); 
    float3 nrm = make_float3(-1.f, 0.f, 0.f ); 

    G4OpBoundaryProcessTest t(pos, mom, pol, nrm) ; 

    unsigned num = OpticksUtil::getenvint("NUM", 10 ); 
    t.propagate_at_boundary(num); 

    return 0 ; 
}


#include <iostream>
#include <iomanip>

#include "G4OpRayleigh.hh"
#include "G4OpticalPhoton.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"

#include "NP.hh"


struct float3 { float x,y,z ;  };
struct float4 { float x,y,z,w ;  };
struct uint4 {  unsigned x,y,z,w ;  };
float3 make_float3(float x, float y, float z)          { float3 v ; v.x = x ; v.y = y ; v.z = z ;           return v ; } 
float4 make_float4(float x, float y, float z, float w ){ float4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
uint4  make_uint4(unsigned x, unsigned y, unsigned z, unsigned w ){ uint4 v ; v.x = x ; v.y = y ; v.z = z ; v.w = w ; return v ; } 
union quad { float4 f ; uint4  u ;  }; 
struct quad4 { quad q0, q1, q2, q3 ; }; 


struct G4OpRayleighTest
{
    static const char* FOLD ; 

    const float3&   pos ; 
    const float3&   mom ; 
    const float3&   pol ; 

    G4OpRayleigh*   proc ; 
    G4Track*        track ; 
    G4Step*         step ;  

    G4OpRayleighTest(const float3& pos, const float3& mom, const float3& pol ); 

    void rayleigh_scatter(unsigned num); 
    void save_p0() const ; 

}; 


G4OpRayleighTest::G4OpRayleighTest(const float3& pos_, const float3& mom_, const float3& pol_)
    :
    pos(pos_),
    mom(mom_),
    pol(pol_),
    proc(new G4OpRayleigh()),
    track(nullptr),
    step(new G4Step)
{
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

    G4Material* material = nullptr ; 

    // G4Track::GetMaterial comes from current step preStepPoint 
    pre->SetMaterial(material); 
    post->SetMaterial(material); 

    G4double step_length = 0. ; 

    step->SetPreStepPoint(pre);
    step->SetPostStepPoint(post);
    step->SetStepLength(step_length);
 
    track->SetStep(step);
}

const char* G4OpRayleighTest::FOLD = "/tmp/G4OpRayleighTest" ; 

void G4OpRayleighTest::rayleigh_scatter(unsigned num)
{
    std::vector<quad4> pp(num) ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        G4VParticleChange* change = proc->PostStepDoIt(*track, *step) ;
        G4ParticleChange* pc = dynamic_cast<G4ParticleChange*>(change);  

        const G4ThreeVector* smom = pc->GetMomentumDirection();
        const G4ThreeVector* spol = pc->GetPolarization();

        std::cout 
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

        quad4 p ; 
        p.q0.f = make_float4( pos.x, pos.y, pos.z, 0.f );   
        p.q1.f = make_float4( smom->x(), smom->y(), smom->z(), 0.f  ) ; 
        p.q2.f = make_float4( spol->x(), spol->y(), spol->z(), 0.f  ) ; 
        p.q3.u = make_uint4(  0u, 0u, 0u, i );   

        pp[i] = p ; 
    }

    NP::Write( FOLD, "p.npy",  (float*)pp.data(), pp.size(), 4, 4 ); 
    save_p0(); 
}

void G4OpRayleighTest::save_p0() const 
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

    G4OpRayleighTest t(pos, mom, pol) ; 
    t.rayleigh_scatter(100000); 

    return 0 ; 
}

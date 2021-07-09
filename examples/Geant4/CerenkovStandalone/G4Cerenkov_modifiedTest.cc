
#include <cassert>
#include <iostream>
#include <iomanip>

#include "G4Cerenkov_modified.hh"
#include "G4Electron.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"

#include "OpticksDebug.hh"
#include "NP.hh"

struct G4Cerenkov_modifiedTest
{
    static const char* FOLD ; 
    static std::string MakeLabel( double BetaInverse, double step_length_ ); 

    NP*                       a ;  
    G4MaterialPropertyVector* rindex ; 
    G4Material*               material ; 
    G4Cerenkov_modified*      proc ; 
    OpticksDebug*             par ; 
    OpticksDebug*             gen ; 

    G4Cerenkov_modifiedTest(const char* path) ;  

    void scanBetaInverse(double v0, double v1, double st) ; 
    void PSDI(double BetaInverse, double step_length );
    void save(const G4VParticleChange* pc, const char* reldir  );

}; 

const char* G4Cerenkov_modifiedTest::FOLD = "/tmp/G4Cerenkov_modifiedTest" ; 

G4Cerenkov_modifiedTest::G4Cerenkov_modifiedTest( const char* path )
    :
    a(OpticksDebug::LoadArray(path)),
    rindex( a ? OpticksDebug::MakeProperty(a) : nullptr),
    material( rindex ? OpticksDebug::MakeMaterial(rindex) : nullptr), 
    proc(new G4Cerenkov_modified()),
    par(new OpticksDebug(8,"Params")),
    gen(new OpticksDebug(8,"GenWavelength"))
{
    std::cout << "loaded from " << path << std::endl ;  

    assert( a ); 
    assert( rindex ); 
    assert( material ); 
    assert( proc ) ; 

    proc->BuildThePhysicsTable() ; 


    proc->par = par ; 
    proc->gen = gen ; 
}


void G4Cerenkov_modifiedTest::scanBetaInverse(double v0, double v1, double st)
{
    G4double charge = 1. ; 
    for(double bi=v0 ; bi <= v1 ; bi += st )
    {
        G4double beta = 1./bi ; 
        G4double averageNumberOfPhotons = proc->GetAverageNumberOfPhotons( charge, beta, material, rindex ); 

        std::cout 
            << " bi " << std::setw(10) << std::fixed << std::setprecision(4) << bi 
            << " averageNumberOfPhotons " << std::setw(10) << std::fixed << std::setprecision(4) <<  averageNumberOfPhotons
            << std::endl 
            ;
    }
}


std::string G4Cerenkov_modifiedTest::MakeLabel( double BetaInverse, double step_length_ )
{
    std::stringstream ss ; 

    ss 
       << "BetaInverse_" << std::fixed << std::setprecision(3) << BetaInverse  << "_" 
       << "step_length_" << std::fixed << std::setprecision(3) << step_length_ << "_"
       ; 

#ifdef SKIP_CONTINUE
    ss << "SKIP_CONTINUE" ;  
#else
    ss << "ASIS" ; 
#endif

    return ss.str(); 
} 


void G4Cerenkov_modifiedTest::PSDI(double BetaInverse, double step_length_)
{
    std::string label = MakeLabel(BetaInverse, step_length_); 
    const char* reldir = label.c_str(); 
    std::cout << "G4Cerenkov_modifiedTest::PSDI [" << label << "]" << std::endl ; 

    G4double en = 1.*MeV ; 

    G4double beta = 1./BetaInverse ; 
    G4double pre_beta = beta ; 
    G4double post_beta = beta ; 
    G4double step_length = step_length_  ; 


    G4ParticleMomentum momentum(0., 0., 1.); 
    G4DynamicParticle* particle = new G4DynamicParticle(G4Electron::Definition(),momentum);
    particle->SetPolarization(0., 0., 1. ); 
    particle->SetKineticEnergy(en); 

    G4ThreeVector position(0., 0., 0.); 
    G4double time(0.); 

    G4Track* track = new G4Track(particle,time,position);

    G4StepPoint* pre = new G4StepPoint ; 
    G4StepPoint* post = new G4StepPoint ; 

    G4ThreeVector pre_position(0., 0., 0.);
    G4ThreeVector post_position(0., 0., 1.);

    pre->SetPosition(pre_position); 
    post->SetPosition(post_position); 

    pre->SetVelocity(pre_beta*c_light); 
    assert( pre->GetBeta() == pre_beta ); 

    post->SetVelocity(post_beta*c_light); 
    assert( post->GetBeta() == post_beta ); 

    // G4Track::GetMaterial comes from current step preStepPoint 
    pre->SetMaterial(material); 
    post->SetMaterial(material); 

    G4Step* step = new G4Step ; 
    step->SetPreStepPoint(pre); 
    step->SetPostStepPoint(post); 
    step->SetStepLength(step_length); 
   
    track->SetStep(step); 

    G4VParticleChange* change = proc->PostStepDoIt(*track, *step) ; 

    assert( change ) ;
    save(change, reldir); 




}


void G4Cerenkov_modifiedTest::save(const G4VParticleChange* pc, const char* reldir )
{
    G4int numberOfSecondaries = pc->GetNumberOfSecondaries();
    std::vector<double> v ;  

    for( G4int i=0 ; i < numberOfSecondaries ; i++)
    {   
        G4Track* track =  pc->GetSecondary(i) ; 

        assert( track->GetParticleDefinition() == G4OpticalPhoton::Definition() );  

        const G4DynamicParticle* aCerenkovPhoton = track->GetDynamicParticle() ;

        const G4ThreeVector& photonMomentum = aCerenkovPhoton->GetMomentumDirection() ;   

        const G4ThreeVector& photonPolarization = aCerenkovPhoton->GetPolarization() ; 

        G4double kineticEnergy = aCerenkovPhoton->GetKineticEnergy() ;   

        G4double wavelength = h_Planck*c_light/kineticEnergy ; 

        const G4ThreeVector& aSecondaryPosition = track->GetPosition() ;

        G4double aSecondaryTime = track->GetGlobalTime() ;

        //G4double weight = track->GetWeight() ; 
        G4double weight = kineticEnergy/eV ;   // temporary switch from debugging

       
        v.push_back(aSecondaryPosition.x()/mm); 
        v.push_back(aSecondaryPosition.y()/mm);
        v.push_back(aSecondaryPosition.z()/mm);
        v.push_back(aSecondaryTime/ns); 

        v.push_back(photonMomentum.x()); 
        v.push_back(photonMomentum.y());
        v.push_back(photonMomentum.z());
        v.push_back(weight);     

        v.push_back(photonPolarization.x());
        v.push_back(photonPolarization.y());
        v.push_back(photonPolarization.z());
        v.push_back(wavelength/nm); 

        v.push_back(0.); 
        v.push_back(0.); 
        v.push_back(0.); 
        v.push_back(0.); 
   }

   unsigned itemsize = 16 ; 
   assert( v.size() % itemsize == 0 ); 
   unsigned num_items = v.size()/itemsize ;  

   std::cout 
       << "[ G4Cerenkov_modifiedTest::save"
       << " numberOfSecondaries " << numberOfSecondaries
       << " num_items " << num_items
       << std::endl 
       ;

   if( num_items > 0 ) 
   {
       // creates reldir if needed
       std::string path = OpticksDebug::prepare_path( FOLD, reldir, "photons.npy" ); 
       std::cout << " saving to " << path << std::endl ; 

       NP::Write( FOLD, reldir, "photons.npy", v.data(), num_items, 4, 4 ); 
       par->write(FOLD, reldir,  2, 4); 
       gen->write(FOLD, reldir,  2, 4); 

       a->save(FOLD, reldir, "RINDEX.npy");  
   }

   std::cout 
       << "] G4Cerenkov_modifiedTest::save"
       << std::endl 
       ;
}


int main(int argc, char** argv)
{
    double BetaInverse = argc > 1 ? std::stod(argv[1]) : 1.5 ;  
    double step_length = argc > 2 ? std::stod(argv[2]) : 100.*1000. ; 
    const char* path = "GScintillatorLib/LS_ori/RINDEX.npy" ; 

    std::cout 
         << " RINDEX path " << path 
         << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse 
         << " step_length " << std::setw(10) << std::fixed << std::setprecision(4) << step_length
         << std::endl 
         ;

    G4Cerenkov_modifiedTest t(path); 
    t.scanBetaInverse(1., 2., 0.1); 

    t.PSDI(BetaInverse, step_length); 

    return 0 ;
}

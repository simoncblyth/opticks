
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "G4Cerenkov_modified.hh"
#include "G4Electron.hh"
#include "G4ParticleMomentum.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4VParticleChange.hh"

#include "OpticksDebug.hh"
#include "OpticksUtil.hh"
#include "OpticksRandom.hh"
#include "NP.hh"

template <typename T>
struct G4Cerenkov_modifiedTest
{
    static const char* FOLD ; 
    static std::string MakeLabel( double BetaInverse, double step_length_, int override_fNumPhotons, long seed, bool precooked ); 

    NP*                       a ;  
    G4MaterialPropertyVector* rindex ; 
    G4Material*               material ; 
    G4Cerenkov_modified*      proc ; 
    OpticksDebug<T>*          par ; 
    OpticksDebug<T>*          gen ;
    NP*                       seq ;  
    NP*                       seqmask ;  
    OpticksRandom*            rnd ; 
    long                      seed ; 

    G4Cerenkov_modifiedTest(const char* rindex_path, const char* random_path, const char* seqmask_path, long seed ) ;  

    double GetAverageNumberOfPhotons(double BetaInverse, double charge, bool s2, double& dt );
    NP* scan_GetAverageNumberOfPhotons(double v0, double v1, unsigned nx) ; 

    void PSDI(double BetaInverse, double step_length, int override_fNumPhotons=-1 );
    void save(const G4VParticleChange* pc, const char* reldir  );

}; 

template <typename T>
const char* G4Cerenkov_modifiedTest<T>::FOLD = "/tmp/G4Cerenkov_modifiedTest" ; 

/**
G4Cerenkov_modifiedTest::LoadRandom
------------------------------------

When the path does not end in ".npy" it is assumed to be a directory path
that contains ".npy" arrays to be concatenated.
**/


template <typename T>
G4Cerenkov_modifiedTest<T>::G4Cerenkov_modifiedTest( const char* rindex_path, const char* random_path, const char* seqmask_path, long seed_ )
    :
    a(OpticksUtil::LoadArray(rindex_path)),
    rindex( a ? OpticksUtil::MakeProperty(a) : nullptr),
    material( rindex ? OpticksUtil::MakeMaterial(rindex) : nullptr), 
    proc(new G4Cerenkov_modified()),
    par(new OpticksDebug<T>(8,"Params")),
    gen(new OpticksDebug<T>(8,"GenWavelength")),
    seq(OpticksUtil::LoadRandom(random_path)),
    seqmask(seqmask_path ? NP::Load(seqmask_path) : nullptr),
    rnd(seq ? new OpticksRandom(seq, seqmask) : nullptr ),
    seed(seed_)
{
    std::cout 
        << "loaded from "
        << " rindex_path " << rindex_path 
        << " random_path " << ( random_path ? random_path : "-" )
        << " seqmask_path " << ( seqmask_path ? seqmask_path : "-" )
        << " seq " << ( seq ? seq->desc() : "-" )
        << " seqmask " << ( seqmask ? seqmask->desc() : "-" )
        << " seed " << seed 
        << std::endl
        ;  

    assert( a ); 
    assert( rindex ); 
    assert( material ); 
    assert( proc ) ; 

    proc->BuildThePhysicsTable() ; 
    proc->par = par ; 
    proc->gen = gen ; 
    proc->rnd = rnd ; 

    if( rnd && seed != 0 )
    {
        std::cout 
            << "G4Cerenkov_modifiedTest::G4Cerenkov_modifiedTest"
            << " FATAL running with input random sequence is incompatible with non-zero seed "
            << " seed " << seed 
            << std::endl 
            ; 
        assert(0);  
    }

    if( rnd == nullptr && seed != 0 )
    {
        std::cout 
            << "G4Cerenkov_modifiedTest::G4Cerenkov_modifiedTest"
            << " setting seed " << seed 
            << std::endl 
            ; 
 
        OpticksRandom::SetSeed(seed);  
    }
}



template <typename T>
double G4Cerenkov_modifiedTest<T>::GetAverageNumberOfPhotons(double BetaInverse, double charge, bool s2, double& dt )
{

    typedef std::chrono::high_resolution_clock::time_point TP ; 
    TP t0 = std::chrono::high_resolution_clock::now() ; 

    G4double beta = 1./BetaInverse ; 
    G4double numPhotons = s2 == false  ? 
                       proc->GetAverageNumberOfPhotons( charge, beta, material, rindex )
                    :
                       proc->GetAverageNumberOfPhotons_s2( charge, beta, material, rindex )
                    ;

    TP t1 = std::chrono::high_resolution_clock::now() ; 
    std::chrono::duration<double> t10 = t1 - t0; 
    dt = t10.count()*1e6 ;    

    return numPhotons ;
}

/**
G4Cerenkov_modifiedTest::scan_GetAverageNumberOfPhotons
-----------------------------------------------------------

For analysis and plotting see ana/ckn.py 

**/

template <typename T>
NP* G4Cerenkov_modifiedTest<T>::scan_GetAverageNumberOfPhotons(double v0, double v1, unsigned nx)
{
    NP* a = NP::Make<double>(nx, 5); 
    double* aa = a->values<double>();   

    G4double charge = 1. ; 
    for(unsigned i=0 ; i < nx ; i++)
    {
        double bi = v0 + (v1-v0)*(double(i)/ double(nx - 1)) ;  

        double dt ; 
        double dt_s2 ; 

        G4double averageNumberOfPhotons = GetAverageNumberOfPhotons( bi, charge, false,   dt ); 
        G4double averageNumberOfPhotons_s2 = GetAverageNumberOfPhotons( bi, charge, true, dt_s2 ); 

        aa[5*i+0] = bi ; 
        aa[5*i+1] = averageNumberOfPhotons ; 
        aa[5*i+2] = averageNumberOfPhotons_s2 ; 
        aa[5*i+3] = dt ; 
        aa[5*i+4] = dt_s2 ; 

        std::cout 
            << " bi " << std::setw(10) << std::fixed << std::setprecision(4) << bi 
            << " averageNumberOfPhotons " << std::setw(10) << std::fixed << std::setprecision(4) <<  averageNumberOfPhotons
            << " averageNumberOfPhotons_s2 " << std::setw(10) << std::fixed << std::setprecision(4) <<  averageNumberOfPhotons_s2
            << " dt " << std::setw(10) << std::fixed << std::setprecision(4) << dt
            << " dt_s2 " << std::setw(10) << std::fixed << std::setprecision(4) << dt_s2 
            << std::endl 
            ;
    }
    return a ; 
}


template <typename T>
std::string G4Cerenkov_modifiedTest<T>::MakeLabel( double BetaInverse, double step_length_, int override_fNumPhotons, long seed, bool precooked  )
{
    std::stringstream ss ; 

    ss << "BetaInverse_" << std::fixed << std::setprecision(3) << BetaInverse  << "_" ; 

    if( override_fNumPhotons <= 0 )
    {
       ss << "step_length_" << std::fixed << std::setprecision(3) << step_length_ << "_" ;
    }
    else
    {
       ss << "override_fNumPhotons_" << override_fNumPhotons << "_" ;
    }

#ifdef SKIP_CONTINUE
    ss << "SKIP_CONTINUE" ;  
#else
    ss << "ASIS" ; 
#endif

#ifdef FLOAT_TEST
    ss << "_FLOAT_TEST" ;  
#else
    ss << "" ; 
#endif

    if( precooked )
    {
        ss << "_PRECOOKED" ; 
    }


    if( seed != 0 )
    {
        ss << "_seed_" << seed << "_" ; 
    }

    return ss.str(); 
} 

/**
G4Cerenkov_modifiedTest::PSDI
--------------------------------

Cooks up the environment of objects (G4DynamicParticle/G4Track/G4StepPoint/G4Step) 
needed to be able to call the Cerenkov PostStepDoIt and then calls it
with the resulting G4VParticleChange Cerenkov photons saved to .npy file.

**/

template <typename T>
void G4Cerenkov_modifiedTest<T>::PSDI(double BetaInverse, double step_length_, int override_fNumPhotons )
{
    if(rnd)
    {
        size_t num_indices = rnd->getNumIndices() ;
        override_fNumPhotons = num_indices ; 

        std::cout 
            << "G4Cerenkov_modifiedTest::PSDI"
            << " rnd seq or seqmask constrains the number of photon indices to "
            << override_fNumPhotons
            << std::endl 
            ;
    }

    bool precooked = rnd != nullptr ; 
    std::string label = MakeLabel(BetaInverse, step_length_ , override_fNumPhotons, seed, precooked ); 
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

    if( override_fNumPhotons > 0 )
    {
        proc->override_fNumPhotons = override_fNumPhotons ;  
    }

    G4VParticleChange* change = proc->PostStepDoIt(*track, *step) ; 

    assert( change ) ;
    save(change, reldir); 
}


template <typename T>
void G4Cerenkov_modifiedTest<T>::save(const G4VParticleChange* pc, const char* reldir )
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
       std::string path = OpticksUtil::prepare_path( FOLD, reldir, "photons.npy" ); 
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

/**
G4Cerenkov_modifiedTest.cc
=============================

For random aligned running prepare 256M PRECOOKED randoms (1M,16,16) 
in 10 files summing to 1GB with::

    cd ~/opticks/qudarap
    TEST=F QSimTest
    
And uncomment the below random_path to use the precooked
const char* random_path = "/tmp/QSimTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000" ; 

Then::

    ./G4Cerenkov_modifiedTest.sh 

**/

int main(int argc, char** argv)
{

    double default_BetaInverse = 1.5 ; 
    double default_step_length = 100.*1000. ;  // (mm) : large to bump up the photon stats
    default_step_length = 100. ; 
    int default_override_fNumPhotons = 1000000 ;  // -ve to use standard depending on step_length

    double BetaInverse = argc > 1 ? std::stod(argv[1]) : default_BetaInverse ;  
    double step_length = argc > 2 ? std::stod(argv[2]) : default_step_length ; 
    int override_fNumPhotons = argc > 3 ? atoi(argv[3]) : default_override_fNumPhotons ; 
    long seed = OpticksUtil::getenvint("SEED", 0 ); 

    const char* rindex_path = "GScintillatorLib/LS_ori/RINDEX.npy" ; 

    //const char* random_path = "/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000" ; 
    const char* random_path = nullptr ; 
    const char* mask_path = nullptr ;   // "/tmp/wavelength_deviant_mask.npy" ; 

    double hc_eVnm = h_Planck*c_light/(eV*nm) ; 
    std::cout 
         << " rindex_path " << rindex_path 
         << " random_path " << ( random_path ? random_path : "-" ) 
         << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse 
         << " step_length " << std::setw(10) << std::fixed << std::setprecision(4) << step_length
         << " override_fNumPhotons " << override_fNumPhotons
         << " hc_eVnm " << std::setw(20) << std::fixed << std::setprecision(10) << hc_eVnm  
         << std::endl 
         ;

    G4Cerenkov_modifiedTest<double> t(rindex_path, random_path, mask_path, seed); 

    //double numPhotons = t.GetAverageNumberOfPhotons(1.5, 1.) ; 

    NP* a = t.scan_GetAverageNumberOfPhotons(1., 2., 1001); 
    a->save("/tmp/G4Cerenkov_modifiedTest/scan_GetAverageNumberOfPhotons.npy"); 
 

    //t.PSDI(BetaInverse, step_length, override_fNumPhotons ); 

    return 0 ;
}

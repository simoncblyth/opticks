
#include <iomanip>
#include "G4PhysicsOrderedFreeVector.hh"
#include "G4MaterialPropertyVector.hh"
#include "X4ScintillationIntegral.hh"

#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NPY.hpp"
#include "PLOG.hh"

const plog::Severity X4ScintillationIntegral::LEVEL = PLOG::EnvLevel("X4ScintillationIntegral", "DEBUG" ); 


G4PhysicsOrderedFreeVector* X4ScintillationIntegral::Integral( const G4MaterialPropertyVector* theFastLightVector )
{
     G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector = new G4PhysicsOrderedFreeVector();

          if (theFastLightVector) { 

               G4double currentIN = (*theFastLightVector)[0];

                if (currentIN >= 0.0) {

                    // Create first (photon energy, Scintillation 
                    // Integral pair  

                    G4double currentPM = theFastLightVector->
                        Energy(0);

                    G4double currentCII = 0.0;

                    aPhysicsOrderedFreeVector->
                        InsertValues(currentPM , currentCII);

                    // Set previous values to current ones prior to loop

                    G4double prevPM  = currentPM;
                    G4double prevCII = currentCII;
                    G4double prevIN  = currentIN;

                    // loop over all (photon energy, intensity)
                    // pairs stored for this material  

                    for(size_t ii = 1;
                              ii < theFastLightVector->GetVectorLength();
                              ++ii)
                    {
                        currentPM = theFastLightVector->Energy(ii);

                        currentIN= (*theFastLightVector)[ii];

                        currentCII = 0.5 * (prevIN + currentIN);

                        currentCII = prevCII +
                            (currentPM - prevPM) * currentCII;

                        aPhysicsOrderedFreeVector->
                            InsertValues(currentPM, currentCII);

                        prevPM  = currentPM;
                        prevCII = currentCII;
                        prevIN  = currentIN;
                    }
               }
            }

    return aPhysicsOrderedFreeVector ; 
}


NPY<double>* X4ScintillationIntegral::CreateWavelengthSamples( const G4PhysicsOrderedFreeVector* ScintillatorIntegral_, G4int num_samples )
{
    G4PhysicsOrderedFreeVector* ScintillatorIntegral = const_cast<G4PhysicsOrderedFreeVector*>(ScintillatorIntegral_) ; 

    double mx = ScintillatorIntegral->GetMaxValue() ;  
    LOG(LEVEL)
        << " ScintillatorIntegral.max*1e9 " << std::fixed << std::setw(10) << std::setprecision(4) << mx*1e9 
        ;

    NPY<double>* wl = NPY<double>::make(num_samples); 
    wl->zero();
    double* v_wl = wl->getValues(); 

    for(G4int i=0 ; i < num_samples ; i++)
    {
        G4double u = G4UniformRand() ; 
        G4double CIIvalue = u*mx;
        G4double sampledEnergy = ScintillatorIntegral->GetEnergy(CIIvalue);

        G4double sampledWavelength_nm = h_Planck*c_light/sampledEnergy/nm ; 

        v_wl[i] = sampledWavelength_nm ;  

        if( i < 100 ) std::cout 
            << " sampledEnergy/eV " << std::fixed << std::setw(10) << std::setprecision(4) << sampledEnergy/eV
            << " sampledWavelength_nm " <<  std::fixed << std::setw(10) << std::setprecision(4) <<  sampledWavelength_nm
            << std::endl 
            ;
    }
    return wl ; 
}



NPY<double>* X4ScintillationIntegral::CreateGeant4InterpolatedInverseCDF( const G4PhysicsOrderedFreeVector* scint_, unsigned num_bins ) // static
{
    G4PhysicsOrderedFreeVector* scint =  const_cast<G4PhysicsOrderedFreeVector*>(scint_) ;  // tut tut : G4 GetMaxValue() GetEnergy() non-const 
    double mx = scint->GetMaxValue() ;   // dataVector.back(); because its **ORDERED** to be increasing on Insert

    NPY<double>* icdf = NPY<double>::make(3, num_bins, 1 );  
    icdf->zero(); 

    int ni = icdf->getShape(0); 
    int nj = icdf->getShape(1); 
    int nk = icdf->getShape(2); 

    assert( ni == 3 ); 
    assert( nk == 1 ); 

    int k = 0 ; 
    int l = 0 ;  

    for(int j=0 ; j < nj ; j++)
    {
        double u_0 = double(j)/double(nj) ; 
        double u_10 = double(j)/double(10*nj) ; 
        double u_90 = 0.9 + double(j)/double(10*nj) ; 

        double energy_0 = scint->GetEnergy( u_0*mx ); 
        double energy_10 = scint->GetEnergy( u_10*mx );
        double energy_90 = scint->GetEnergy( u_90*mx );
 
        double wavelength_0 = h_Planck*c_light/energy_0/nm ;
        double wavelength_10 = h_Planck*c_light/energy_10/nm ;
        double wavelength_90 = h_Planck*c_light/energy_90/nm ;

        bool chk = false ; 
        icdf->setValue(0, j, k, l,  chk ? u_0   : wavelength_0 ); 
        icdf->setValue(1, j, k, l,  chk ? u_10  : wavelength_10 ); 
        icdf->setValue(2, j, k, l,  chk ? u_90  : wavelength_90 ); 
    }
    return icdf ; 
} 



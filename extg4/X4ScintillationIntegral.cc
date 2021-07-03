
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


/**
X4ScintillationIntegral::CreateGeant4InterpolatedInverseCDF
-------------------------------------------------------------

Reproducing the results of Geant4 dynamic bin finding interpolation 
using GPU texture lookups demands very high resolution textures for some 
ICDF shapes. This function prepares a three item buffer that can be used
to create a 2D texture that effectively mimmicks variable bin sizing even 
though GPU hardware does not support that without paying the cost of
high resolution across the entire range.

* item 0 : full range "standard" resolution
* item 1: left hand side high resolution 
* item 2: right hand side high resolution 

::

    hd_factor                LHS            RHS
    10          10x bins:    0.00->0.10     0.90->1.00 
    20          20x bins:    0.00->0.05     0.95->1.00 

**/

NPY<double>* X4ScintillationIntegral::CreateGeant4InterpolatedInverseCDF( 
       const G4PhysicsOrderedFreeVector* ScintillatorIntegral_, 
       unsigned num_bins, 
       unsigned hd_factor, 
       const char* name
) 
{
    G4PhysicsOrderedFreeVector* ScintillatorIntegral = const_cast<G4PhysicsOrderedFreeVector*>(ScintillatorIntegral_) ;  // tut tut : G4 GetMaxValue() GetEnergy() non-const 
    double mx = ScintillatorIntegral->GetMaxValue() ;   // dataVector.back(); because its **ORDERED** to be increasing on Insert

    NPY<double>* icdf = NPY<double>::make(3, num_bins, 1 );  
    icdf->zero(); 

    int ni = icdf->getShape(0); 
    int nj = icdf->getShape(1); 
    int nk = icdf->getShape(2); 

    assert( ni == 3 ); 
    assert( nk == 1 ); 

    int k = 0 ; 
    int l = 0 ;  

    assert( hd_factor == 10 || hd_factor == 20 ); 
    double edge = 1./double(hd_factor) ;  

    icdf->setMeta<std::string>("name", name ); 
    icdf->setMeta<std::string>("creator", "X4ScintillationIntegral::CreateGeant4InterpolatedInverseCDF" ); 
    icdf->setMeta<int>("hd_factor", hd_factor ); 
    icdf->setMeta<int>("num_bins", num_bins ); 
    icdf->setMeta<double>("edge", edge ); 


    LOG(LEVEL) 
        << " num_bins " << num_bins
        << " hd_factor " << hd_factor
        << " mx " << std::fixed << std::setw(10) << std::setprecision(4) << mx
        << " mx*1e9 " << std::fixed << std::setw(10) << std::setprecision(4) << mx*1e9
        << " edge " << std::fixed << std::setw(10) << std::setprecision(4) << edge 
        << " icdf " << icdf->getShapeString()
        ;

    for(int j=0 ; j < nj ; j++)
    {
        double u_all = double(j)/double(nj) ;                         // 0 -> (nj-1)/nj = 1-1/nj
        double u_lhs = double(j)/double(hd_factor*nj) ;              
        double u_rhs = 1. - edge + double(j)/double(hd_factor*nj) ; 

        double energy_all = ScintillatorIntegral->GetEnergy( u_all*mx ); 
        double energy_lhs = ScintillatorIntegral->GetEnergy( u_lhs*mx );
        double energy_rhs = ScintillatorIntegral->GetEnergy( u_rhs*mx );
 
        double wavelength_all = h_Planck*c_light/energy_all/nm ;
        double wavelength_lhs = h_Planck*c_light/energy_lhs/nm ;
        double wavelength_rhs = h_Planck*c_light/energy_rhs/nm ;

        bool chk_dom = false ; 
        icdf->setValue(0, j, k, l,  chk_dom ? u_all : wavelength_all ); 
        icdf->setValue(1, j, k, l,  chk_dom ? u_lhs : wavelength_lhs ); 
        icdf->setValue(2, j, k, l,  chk_dom ? u_rhs : wavelength_rhs ); 
    }
    return icdf ; 
} 



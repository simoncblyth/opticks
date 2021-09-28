
#include <iomanip>
#include "X4PhysicsOrderedFreeVector.hh"
#include "G4MaterialPropertyVector.hh"
#include "X4Scintillation.hh"
#include "X4MaterialPropertyVector.hh"

#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NPY.hpp"
#include "PLOG.hh"

const plog::Severity X4Scintillation::LEVEL = PLOG::EnvLevel("X4Scintillation", "DEBUG" ); 


X4Scintillation::X4Scintillation( const NPY<double>* fast_, const NPY<double>* slow_ )
    :
    fast(fast_),
    slow(slow_),
    epsilon(0.), 
    mismatch(NPY<double>::compare(fast, slow, epsilon, true, 100, 'A')), 
    theFastLightVector(X4MaterialPropertyVector::FromArray(fast)),
    theSlowLightVector(X4MaterialPropertyVector::FromArray(slow)),
    ScintillationIntegral(Integral(theFastLightVector))
{
    assert( mismatch == 0 ); 
}


NPY<double>* X4Scintillation::createWavelengthSamples( unsigned num_samples )
{
    return CreateWavelengthSamples(ScintillationIntegral, num_samples ); 
} 
NPY<double>* X4Scintillation::createGeant4InterpolatedInverseCDF( unsigned num_bins, unsigned hd_factor, const char* material_name, bool energy_not_wavelength  )
{
    return CreateGeant4InterpolatedInverseCDF( ScintillationIntegral, num_bins, hd_factor, material_name, energy_not_wavelength  ); 
}


/**
X4Scintillation::Integral
---------------------------

Returns cumulative sum of the input property on the same energy domain, 
with values starting at 0. and increasing monotonically.

The is using trapezoidal numerical integration.


**/

G4MaterialPropertyVector* X4Scintillation::Integral( const G4MaterialPropertyVector* theFastLightVector )
{
     G4MaterialPropertyVector* aMaterialPropertyVector = new G4MaterialPropertyVector();

          if (theFastLightVector) { 

               G4double currentIN = (*theFastLightVector)[0];

                if (currentIN >= 0.0) {

                    // Create first (photon energy, Scintillation 
                    // Integral pair  

                    G4double currentPM = theFastLightVector->
                        Energy(0);

                    G4double currentCII = 0.0;

                    aMaterialPropertyVector->
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

                        aMaterialPropertyVector->
                            InsertValues(currentPM, currentCII);

                        prevPM  = currentPM;
                        prevCII = currentCII;
                        prevIN  = currentIN;
                    }
               }
            }

    return aMaterialPropertyVector ; 
}


NPY<double>* X4Scintillation::CreateWavelengthSamples( const G4MaterialPropertyVector* ScintillatorIntegral_, unsigned num_samples )
{
    G4MaterialPropertyVector* ScintillatorIntegral = const_cast<G4MaterialPropertyVector*>(ScintillatorIntegral_) ; 

    double mx = ScintillatorIntegral->GetMaxValue() ;  
    LOG(LEVEL)
        << " ScintillatorIntegral.max*1e9 " << std::fixed << std::setw(10) << std::setprecision(4) << mx*1e9 
        ;

    NPY<double>* wl = NPY<double>::make(num_samples); 
    wl->zero();
    double* v_wl = wl->getValues(); 

    for(unsigned i=0 ; i < num_samples ; i++)
    {
        G4double u = G4UniformRand() ; 
        G4double CIIvalue = u*mx;
        G4double sampledEnergy = ScintillatorIntegral->GetEnergy(CIIvalue);  // from value to domain 

        G4double sampledWavelength_nm = h_Planck*c_light/sampledEnergy/nm ; 

        v_wl[i] = sampledWavelength_nm ;  

        if( i < 10 ) std::cout 
            << " sampledEnergy/eV " << std::fixed << std::setw(10) << std::setprecision(4) << sampledEnergy/eV
            << " sampledWavelength_nm " <<  std::fixed << std::setw(10) << std::setprecision(4) <<  sampledWavelength_nm
            << std::endl 
            ;
    }
    return wl ; 
}


/**
X4Scintillation::CreateGeant4InterpolatedInverseCDF
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


The ICDF is formed using Geant4s "domain lookup from value" functionality 
in the form of G4MaterialPropertyVector::GetEnergy 

::

    g4-cls G4MaterialPropertyVector

    096 G4double G4PhysicsOrderedFreeVector::GetEnergy(G4double aValue)
     97 {
     98         G4double e;
     99         if (aValue <= GetMinValue()) {
    100           e = edgeMin;
    101         } else if (aValue >= GetMaxValue()) {
    102           e = edgeMax;
    103         } else {
    104           size_t closestBin = FindValueBinLocation(aValue);
    105           e = LinearInterpolationOfEnergy(aValue, closestBin);
    106     }
    107         return e;
    108 }

    118 G4double G4PhysicsOrderedFreeVector::LinearInterpolationOfEnergy(G4double aValue,
    119                                  size_t bin)
    120 {
    121         G4double res = binVector[bin];
    122         G4double del = dataVector[bin+1] - dataVector[bin];
    123         if(del > 0.0) { 
    124           res += (aValue - dataVector[bin])*(binVector[bin+1] - res)/del;
    125         }
    126         return res;
    127 }



   
                                                        1  (x1,y1)     (  binVector[bin+1], dataVector[bin+1] )
                                                       /
                                                      /
                                                     *  ( xv,yv )       ( res, aValue )      
                                                    /
                                                   /
                                                  0  (x0,y0)          (  binVector[bin], dataVector[bin] )


              Similar triangles::
               
                 xv - x0       x1 - x0 
               ---------- =   -----------
                 yv - y0       y1 - y0 




                  res - binVector[bin]             binVector[bin+1] - binVector[bin]
               ----------------------------  =     -----------------------------------
                 aValue - dataVector[bin]          dataVector[bin+1] - dataVector[bin] 


                                                                              binVector[bin+1] - binVector[bin]
                   res  = binVector[bin] +  ( aValue - dataVector[bin] ) *  -------------------------------------
                                                                              dataVector[bin+1] - dataVector[bin] 

                                                   x1 - x0
                   xv  =    x0  +   (yv - y0) *  -------------- 
                                                   y1 - y0

**/

NPY<double>* X4Scintillation::CreateGeant4InterpolatedInverseCDF( 
       const G4MaterialPropertyVector* ScintillatorIntegral_, 
       unsigned num_bins, 
       unsigned hd_factor, 
       const char* material_name, 
       bool energy_not_wavelength
)   // static
{
    G4MaterialPropertyVector* ScintillatorIntegral = const_cast<G4MaterialPropertyVector*>(ScintillatorIntegral_) ;  // tut tut : G4 GetMaxValue() GetEnergy() non-const 
    double mx = ScintillatorIntegral->GetMaxValue() ;   // dataVector.back(); because its **ORDERED** to be increasing on Insert


    // hmm more extensible (eg for Cerenkov [BetaInverse,u,payload] icdf)  
    // with the 3 for the different resolutions to be in the payload rather than as separate items ?
    // would of course use 4 to map to float4 after narrowing 

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

    if(material_name)
    {
        icdf->setMeta<std::string>("name", material_name ); 
    }

    icdf->setMeta<std::string>("creator", "X4Scintillation::CreateGeant4InterpolatedInverseCDF" ); 
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
        double u_lhs = double(j)/double(hd_factor*nj) ;               // hd_factor=10(20) u_lhs=0.0->0.1  (0.0  ->~0.05) 
        double u_rhs = 1. - edge + double(j)/double(hd_factor*nj) ;   // hd_factor=10(20) u_rhs=0.9->~1.0 (0.95 ->~1.0)
        // u_lhs and u_rhs mean there are hd_factor more lookups at the extremes 

        double energy_all = ScintillatorIntegral->GetEnergy( u_all*mx ); 
        double energy_lhs = ScintillatorIntegral->GetEnergy( u_lhs*mx );
        double energy_rhs = ScintillatorIntegral->GetEnergy( u_rhs*mx );
 
        double wavelength_all = h_Planck*c_light/energy_all/nm ;
        double wavelength_lhs = h_Planck*c_light/energy_lhs/nm ;
        double wavelength_rhs = h_Planck*c_light/energy_rhs/nm ;

        icdf->setValue(0, j, k, l,  energy_not_wavelength ? energy_all :  wavelength_all ); 
        icdf->setValue(1, j, k, l,  energy_not_wavelength ? energy_lhs :  wavelength_lhs ); 
        icdf->setValue(2, j, k, l,  energy_not_wavelength ? energy_rhs :  wavelength_rhs ); 
    }
    return icdf ; 
} 



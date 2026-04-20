#pragma once
/**
U4ScintCommon
==============

Collect here functionality used from both U4Scint.h and U4ScintThree.h

**/

struct U4ScintCommon
{
    static constexpr const bool VERBOSE = false ;

    static G4MaterialPropertyVector* Integral( const G4MaterialPropertyVector* theFastLightVector ) ;

    template<typename T>
    static NP* CreateWavelengthSamples(
        const G4MaterialPropertyVector* ScintillatorIntegral,
        size_t num_samples
        );

    static NP* CreateGeant4InterpolatedInverseCDF(
        const G4MaterialPropertyVector* ScintillatorIntegral,
        int num_bins,
        int hd_factor,
        const char* name,
        bool energy_not_wavelength
        );

};



/**
U4ScintCommon::Integral
---------------------------

Returns cumulative sum of the input property on the same energy domain,
with values starting at 0. and increasing monotonically.

The is using trapezoidal numerical integration.


**/

inline G4MaterialPropertyVector* U4ScintCommon::Integral( const G4MaterialPropertyVector* theFastLightVector ) // static
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


/**
U4ScintCommon::CreateWavelengthSamples
---------------------------------------

**/

template<typename T>
inline NP* U4ScintCommon::CreateWavelengthSamples(
    const G4MaterialPropertyVector* ScintillatorIntegral_cdf_,
    size_t num_samples )
{
    if(num_samples == 0) return nullptr ;

    G4MaterialPropertyVector* ScintillatorIntegral_cdf = const_cast<G4MaterialPropertyVector*>(ScintillatorIntegral_cdf_) ;

    double mx = ScintillatorIntegral_cdf->GetMaxValue() ;
    double mn = ScintillatorIntegral_cdf->GetMinValue() ;
    if(VERBOSE) std::cerr
        << "U4ScintCommon::CreateWavelengthSamples"
        << " ScintillatorIntegral_cdf.max*1e9 "
        << std::fixed << std::setw(10) << std::setprecision(4) << mx*1e9
        << " ScintillatorIntegral_cdf.min*1e9 "
        << std::fixed << std::setw(10) << std::setprecision(4) << mn*1e9
        ;

    NP* w = NP::Make<T>(num_samples);
    w->fill<T>(0.);
    T* ww = w->values<T>() ;



    for(size_t i=0 ; i < num_samples ; i++)
    {
        G4double u = G4UniformRand() ;
        //G4double CIIvalue = u*(mx-mn);  // as its a CDF mn will be zero
        G4double CIIvalue = u*mx;
        G4double sampledEnergy = ScintillatorIntegral_cdf->GetEnergy(CIIvalue);  // from value to domain

        G4double sampledWavelength_nm = h_Planck*c_light/sampledEnergy/nm ;

        ww[i] = sampledWavelength_nm ;

        if( VERBOSE && i < 10 ) std::cout
            << " sampledEnergy/eV "
            << std::fixed << std::setw(10) << std::setprecision(4) << sampledEnergy/eV
            << " sampledWavelength_nm "
            <<  std::fixed << std::setw(10) << std::setprecision(4) << sampledWavelength_nm
            << std::endl
            ;
    }
    return w ;
}


/**
U4ScintCommon::CreateGeant4InterpolatedInverseCDF
-----------------------------------------------------

Reproducing the results of Geant4 dynamic bin finding interpolation
using GPU texture lookups demands very high resolution textures for some
ICDF shapes. This function prepares a three item buffer that can be used
to create a 2D texture that effectively mimmicks variable bin sizing even
though GPU hardware does not support that without paying the cost of
high resolution across the entire range.

* item 0: full range "standard" resolution
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

inline NP* U4ScintCommon::CreateGeant4InterpolatedInverseCDF(
       const G4MaterialPropertyVector* ScintillatorIntegral_,
       int num_bins,
       int hd_factor,
       const char* material_name,
       bool energy_not_wavelength
)   // static
{

    assert( material_name );

    G4MaterialPropertyVector* ScintillatorIntegral = const_cast<G4MaterialPropertyVector*>(ScintillatorIntegral_) ;  // tut tut : G4 GetMaxValue() GetEnergy() non-const
    double mx = ScintillatorIntegral->GetMaxValue() ;   // dataVector.back(); because its **ORDERED** to be increasing on Insert


    // hmm more extensible (eg for Cerenkov [BetaInverse,u,payload] icdf)
    // with the 3 for the different resolutions to be in the payload rather than as separate items ?
    // would of course use 4 to map to float4 after narrowing


    NP* icdf = NP::Make<double>(3, num_bins, 1);
    icdf->fill<double>(0.);

    int ni = icdf->shape[0];
    int nj = icdf->shape[1];
    int nk = icdf->shape[2];

    assert( ni == 3 );
    assert( nk == 1 );
    int k = 0 ;

    assert( hd_factor == 10 || hd_factor == 20 );
    double edge = 1./double(hd_factor) ;
    double margin = edge/10. ;
    //  HD20 edge 0.05 margin 0.005


    icdf->names.push_back(material_name) ;  // match X4/GGeo
    icdf->set_meta<std::string>("name", material_name );

    icdf->set_meta<std::string>("creator", "U4ScintCommon::CreateGeant4InterpolatedInverseCDF" );
    icdf->set_meta<int>("hd_factor", hd_factor );
    icdf->set_meta<int>("num_bins", num_bins );
    icdf->set_meta<double>("edge", edge );
    icdf->set_meta<double>("margin", margin );


    if(VERBOSE) std::cerr
        << "U4ScintCommon::CreateGeant4InterpolatedInverseCDF"
        << " num_bins " << num_bins
        << " hd_factor " << hd_factor
        << " mx " << std::fixed << std::setw(10) << std::setprecision(4) << mx
        << " mx*1e9 " << std::fixed << std::setw(10) << std::setprecision(4) << mx*1e9
        << " edge " << std::fixed << std::setw(10) << std::setprecision(4) << edge
        << " margin " << std::fixed << std::setw(10) << std::setprecision(4) << margin
        << " icdf " << icdf->sstr()
        << std::endl
        ;



    for(int j=0 ; j < nj ; j++)
    {
        double frac = double(j)/double(nj-1);
        double u_all, u_lhs, u_mid, u_rhs ;

        if( margin == 0. )
        {
            u_all = frac ;
            u_lhs = frac*edge ;
            u_mid = edge + frac*(1.0-2.0*edge) ;
            u_rhs = 1. - edge + frac*edge ;
        }
        else if( margin > 0. )
        {
             u_all = frac ;                                                   // ALL : 0->1
             u_lhs = frac*(edge + margin) ;                                   // LHS : Map   0 -> e+m
             u_mid = (edge - margin) + frac*(1.0 - 2.0*edge + 2.0*margin ) ;  // MID : Map (e-m) -> (1-e+m)
             u_rhs = (1. - edge - margin) + frac*(edge + margin) ;            // RHS : Map  ( 1.0 - edge - margin)  -> 1
        }

/**

HD20  e = 0.05
     2e = 0.1
      m = 0.005
     2m = 0.01



         0               e-m   e   e+m                                  1-e-m 1-e 1-e+m                    1
         |                |    |    |                                     |    |    |                      |

         +-------LHS----------------+                                     +----------RHS-------------------+
                          +---------------------MID---------------------------------+



                          |    1 - e + m - (e - m)  = 1 - 2*e + 2m                  |
                                                      1 - 0.1 + 0.01 = 0.91         |
                          |   edges trimmed with margins added                      |



    u0 < e-m : LHS-only region
       x = u0 / (e+m )     ## NB using actual range of the LHS texture for the mapping

    u0 [e-m -> e+m] : LHS+MID overlap region

       x =




**/


        //double energy_all = ScintillatorIntegral->GetEnergy( u_all*mx );
        double energy_mid = ScintillatorIntegral->GetEnergy( u_mid*mx );
        double energy_lhs = ScintillatorIntegral->GetEnergy( u_lhs*mx );
        double energy_rhs = ScintillatorIntegral->GetEnergy( u_rhs*mx );

        //double wavelength_all = h_Planck*c_light/energy_all/nm ;
        double wavelength_mid = h_Planck*c_light/energy_mid/nm ;
        double wavelength_lhs = h_Planck*c_light/energy_lhs/nm ;
        double wavelength_rhs = h_Planck*c_light/energy_rhs/nm ;

        //double v_all = energy_not_wavelength ? energy_all :  wavelength_all ;
        double v_mid = energy_not_wavelength ? energy_mid :  wavelength_mid ;
        double v_lhs = energy_not_wavelength ? energy_lhs :  wavelength_lhs ;
        double v_rhs = energy_not_wavelength ? energy_rhs :  wavelength_rhs ;

        //icdf->set<double>(v_all, 0, j, k );
        icdf->set<double>(v_mid, 0, j, k );
        icdf->set<double>(v_lhs, 1, j, k );
        icdf->set<double>(v_rhs, 2, j, k );
    }
    return icdf ;
}




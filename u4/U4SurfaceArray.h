#pragma once 
/**
U4SurfaceArray.h 
==================

Formerly did this in U4Surface::MakeStandardArray

* this is a reimplementation that follows the intent of GSurfaceLib::createStandardSurface
* for surfaces only payload group zero is filled with four payload
  probabilities that sum to one::

    (detect, absorb, reflect_specular, reflect_diffuse)

* EFFICIENCY, REFLECTIVITY and specular/diffuse optical surface nature 
  are inputs to setting the four probabilities that must sum to 1. 

* notice that implicit surfaces are like perfects in that there 
  is no wavelength variation of properties : the implicits 
  always just perfectly absorb corresponding to Geant4 fStopAndKill
  behaviour when going from RINDEX to non-RINDEX material. 


Only simple surfaces are covered by this
------------------------------------------

U4SurfaceArray currently only handles simple surfaces where each surface
is either:

* sensor
* absorber
* specular refector 
* diffuse reflector

As implemented in qsim::propagate_at_surface

Other types of surfaces need to use a different approach and 
steered to the appropriate qsim method by qsim::propagate 
based on the "ems" or smatsur.h enum which is planted 
into the optical buffer by sstandard::make_optical


**/

#include <vector>
#include <string>

#include "G4SystemOfUnits.hh"
#include "G4LogicalSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"

#include "NP.hh"
#include "sdomain.h"
#include "sprop.h"

#include "U4SurfacePerfect.h"
#include "U4OpticalSurfaceFinish.h"

struct U4SurfaceArray
{
    static constexpr const bool VERBOSE = false ; 
    static constexpr const double UNSET = -1. ; 
    struct D4 { double x,y,z,w ; } ;

    sdomain dom ; 
    int num_surface ; 
    int num_implicit ; 
    int num_perfect ; 
    int ni ; 
    int nj ; 
    int nk ; 
    int nl ; 
    int j  ;  // 0:payload group 

    NP* sur ; 
    double* sur_v ; 

    U4SurfaceArray(
        const std::vector<const G4LogicalSurface*>& surface, 
        const std::vector<std::string>& implicit, 
        const std::vector<U4SurfacePerfect>& perfect  
        ); 

    void addSurface( int i, const G4LogicalSurface* ls); 
    void addImplicit(int i, const char* name); 
    void addPerfect( int i, const U4SurfacePerfect& perfect ); 
};


inline U4SurfaceArray::U4SurfaceArray(
        const std::vector<const G4LogicalSurface*>& surface, 
        const std::vector<std::string>& implicit, 
        const std::vector<U4SurfacePerfect>& perfect  )
    :
    num_surface(surface.size()),
    num_implicit(implicit.size()),
    num_perfect(perfect.size()),
    ni(num_surface + num_implicit + num_perfect),
    nj(sprop::NUM_PAYLOAD_GRP),
    nk(dom.length),
    nl(sprop::NUM_PAYLOAD_VAL),
    j(0),   // payload group
    sur(NP::Make<double>(ni, nj, nk, nl )),
    sur_v(sur->values<double>()) 
{
    sur->fill<double>(UNSET); // matching X4/GGeo 

    for(int i=0 ; i < ni ; i++)
    {
        if( i < num_surface )
        {
            const G4LogicalSurface* ls = surface[i] ; 
            addSurface(i, ls); 
        }
        else if( i < num_surface + num_implicit )
        {
            const std::string& impl = implicit[i-num_surface] ; 
            addImplicit(i, impl.c_str()); 
        }
        else if( i < num_surface + num_implicit + num_perfect )
        {
            const U4SurfacePerfect& perf = perfect[i-num_surface-num_implicit] ; 
            addPerfect(i, perf); 
        }
    }
    assert( int(sur->names.size()) == ni ); 
}




/**
U4SurfaceArray::addSurface
----------------------------

Payload values::

   detect
   absorb 
   reflect_specular 
   reflect_diffuse

depend on REFLECTIVITY and EFFICIENCY properties. 
BUT, there is not a one-to-one relationship between 
the properties and the content of the sur array. 

Rather there is dependency on the surface being a sensor 
(having any efficiency greater than zero) and also 
the U4OpticalSurfaceFinish::IsPolished specular nature.
Those inputs determine what is filled into the payload 
slots that all add to one and are treated as probabilities. 

Note assumption that sensors do not reflect, this is old traditional POM 
for more involved modelling need to use special surfaces, see smatsur.h. 
 

**/

inline void U4SurfaceArray::addSurface(int i, const G4LogicalSurface* ls)
{
    const G4String& name = ls->GetName() ; 
    sur->names.push_back(name.c_str()) ; 

    G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(ls->GetSurfaceProperty());
    unsigned finish = os->GetFinish() ;
    bool is_specular = U4OpticalSurfaceFinish::IsPolished( finish ); 

    G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
    assert( mpt ); 
    //if( mpt == nullptr ) std::cerr << "U4Surface::MakeStandardArray NO MPT " << name << std::endl ; 

    G4MaterialPropertyVector* EFFICIENCY = mpt ? mpt->GetProperty("EFFICIENCY") : nullptr ; 
    G4MaterialPropertyVector* REFLECTIVITY = mpt ? mpt->GetProperty("REFLECTIVITY") : nullptr ; 

    double max_effi = 0. ; 
    double max_refl = 0. ; 

    for(int k=0 ; k < nk ; k++)  // energy/wavelength domain 
    {
        double energy = dom.energy_eV[k] * eV ; 
        double effi = EFFICIENCY ? EFFICIENCY->Value(energy) : 0. ; 
        double refl = REFLECTIVITY ? REFLECTIVITY->Value(energy) : 0. ; 
        if( effi > max_effi ) max_effi = effi ; 
        if( refl > max_refl ) max_refl = refl ; 
    }
        
    bool is_sensor = max_effi > 0. ; 

    if(VERBOSE) std::cout 
        << "U4SurfaceArray::addSurface"
        << " name " << std::setw(30) << name
        << " max_effi " << std::setw(10) << std::fixed << std::setprecision(4) << max_effi
        << " max_refl " << std::setw(10) << std::fixed << std::setprecision(4) << max_refl
        << " is_sensor " << ( is_sensor ? "YES" : "NO " )
        << " is_specular " << ( is_specular ? "YES" : "NO " )
        << " finish " << U4OpticalSurfaceFinish::Name(finish) 
        << std::endl 
        ;

    for(int k=0 ; k < nk ; k++)  // energy/wavelength domain 
    {
        double energy = dom.energy_eV[k] * eV ; 
        double effi = EFFICIENCY ? EFFICIENCY->Value(energy) : 0. ; 
        double refl = REFLECTIVITY ? REFLECTIVITY->Value(energy) : 0. ; 

       D4 d4 ; 

        if( is_sensor )   
        {
            d4.x = effi ;       // detect
            d4.y = 1. - effi ;  // absorb 
            d4.z = 0. ;         // reflect_specular 
            d4.w = 0. ;         // reflect_diffuse
        } 
        else 
        {
            if( is_specular )
            {
                d4.x = 0. ;          // detect 
                d4.y = 1. - refl  ;  // absorb  
                d4.z = refl ;        // reflect_specular            
                d4.w = 0. ;          // reflect_diffuse 
            }
            else
            {
                d4.x = 0. ;          // detect 
                d4.y = 1. - refl  ;  // absorb  
                d4.z = 0. ;          // reflect_specular            
                d4.w = refl ;        // reflect_diffuse 
            }
        }
        double d4_sum = d4.x + d4.y + d4.z + d4.w ; 
        assert( std::abs( d4_sum - 1. ) < 1e-9 ); 
        int index = i*nj*nk*nl + j*nk*nl + k*nl ;
        sur_v[index+0] = d4.x ;
        sur_v[index+1] = d4.y ;
        sur_v[index+2] = d4.z ;
        sur_v[index+3] = d4.w ;
    }
}

/**
U4SurfaceArray::addImplicit
----------------------------

Implicits are perfect absorbers that mimic within Opticks on GPU
the implicit fStopAndKill absorption behaviour of Geant4 when 
photons reach a boundary from a material with RINDEX property
to a material without RINDEX property.
Typically this is photons hitting non-transparent materials. 

**/

inline void U4SurfaceArray::addImplicit(int i, const char* name)
{
    U4SurfacePerfect impl = { name, 0., 1., 0., 0. } ; 
    addPerfect(i, impl ); 
}

/**
U4SurfaceArray::addPerfect
----------------------------

Perfect surfaces are used for debugging and unrealistic tests.  
They have constant properties across the energy/wavelength domain, 
implemented by simply copying the values. 

**/

inline void U4SurfaceArray::addPerfect(int i, const U4SurfacePerfect& perf )
{
    sur->names.push_back(perf.name.c_str()) ; 
    for(int k=0 ; k < nk ; k++)  // energy/wavelength domain 
    {
        assert( std::abs( perf.sum() - 1. ) < 1e-9 ); 
        int index = i*nj*nk*nl + j*nk*nl + k*nl ;
        sur_v[index+0] = perf.detect ;
        sur_v[index+1] = perf.absorb ;
        sur_v[index+2] = perf.reflect_specular ;
        sur_v[index+3] = perf.reflect_diffuse ;
    }
}


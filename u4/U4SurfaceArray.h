#pragma once 
/**
U4SurfaceArray.h 
==================

Formerly did this in U4Surface::MakeStandardArray

* needs to follow the intent of GSurfaceLib::createStandardSurface
* for surfaces only payload group zero is filled with four payload
  probabilities that sum to one::

    (detect, absorb, reflect_specular, reflect_diffuse)

* EFFICIENCY, REFLECTIVITY and specular/diffuse optical surface nature 
  are inputs to setting the four probabilities that must sum to 1. 


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
    struct D4 { double x,y,z,w ; } ;

    sdomain dom ; 
    int num_surfaces ; 
    int num_implicits ; 
    int num_perfects ; 
    int ni ; 
    int nj ; 
    int nk ; 
    int nl ; 
    int j  ;  // payload group 

    NP* sur ; 
    double* sur_v ; 
    std::vector<std::string> names ; 

    U4SurfaceArray(
        const std::vector<const G4LogicalSurface*>& surfaces, 
        const std::vector<std::string>& implicits, 
        const std::vector<U4SurfacePerfect>& perfects  ); 

    void addSurface( int i, const G4LogicalSurface* ls); 
    void addImplicit(int i, const char* name); 
    void addPerfect( int i, const U4SurfacePerfect& perfect ); 
};




inline U4SurfaceArray::U4SurfaceArray(
        const std::vector<const G4LogicalSurface*>& surfaces, 
        const std::vector<std::string>& implicits, 
        const std::vector<U4SurfacePerfect>& perfects  )
    :
    num_surfaces(surfaces.size()),
    num_implicits(implicits.size()),
    num_perfects(perfects.size()),
    ni(num_surfaces + num_implicits + num_perfects),
    nj(sprop::NUM_PAYLOAD_GRP),
    nk(dom.length),
    nl(sprop::NUM_PAYLOAD_VAL),
    j(0),   // payload group
    sur(NP::Make<double>(ni, nj, nk, nl )),
    sur_v(sur->values<double>()) 
{
    for(int i=0 ; i < ni ; i++)
    {
        if( i < num_surfaces )
        {
            const G4LogicalSurface* ls = surfaces[i] ; 
            addSurface(i, ls); 
        }
        else if( i < num_surfaces + num_implicits )
        {
            const std::string& implicit = implicits[i-num_surfaces] ; 
            addImplicit(i, implicit.c_str()); 
        }
        else if( i < num_surfaces + num_implicits + num_perfects )
        {
            const U4SurfacePerfect& perfect = perfects[i-num_surfaces-num_implicits] ; 
            addPerfect(i, perfect); 
        }
    }
    sur->set_names(names) ; 
}

inline void U4SurfaceArray::addSurface(int i, const G4LogicalSurface* ls)
{
    const G4String& name = ls->GetName() ; 
    names.push_back(name.c_str()) ; 

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

    std::cout 
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

        // note assumption that sensors dont reflect, this is old traditional POM 
        // for more involved modelling need to use special surfaces 
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
the implicit fStopAndKill absorption that Geant4 does when photons 
are on boundary of a material with RINDEX and one without RINDEX.  

**/

inline void U4SurfaceArray::addImplicit(int i, const char* name)
{
    U4SurfacePerfect implicit = { name, 0., 1., 0., 0. } ; 
    addPerfect(i, implicit); 
}
inline void U4SurfaceArray::addPerfect(int i, const U4SurfacePerfect& perfect )
{
    names.push_back(perfect.name.c_str()) ; 
    for(int k=0 ; k < nk ; k++)  // energy/wavelength domain 
    {
        assert( std::abs( perfect.sum() - 1. ) < 1e-9 ); 
        int index = i*nj*nk*nl + j*nk*nl + k*nl ;
        sur_v[index+0] = perfect.detect ;
        sur_v[index+1] = perfect.absorb ;
        sur_v[index+2] = perfect.reflect_specular ;
        sur_v[index+3] = perfect.reflect_diffuse ;
    }
}


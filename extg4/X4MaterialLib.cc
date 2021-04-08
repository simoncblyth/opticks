/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include "G4Material.hh"
#include "X4MaterialLib.hh"
#include "X4PropertyMap.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "PLOG.hh"

/**
X4MaterialLib::Standardize
----------------------------

* requires: both Geant4 G4MaterialTable and Opticks GMaterialLib 

* must be same number/names/order of the materials from both 

* for Geant4 materials with MPT (G4MaterialPropertiesTable) replaces it
  with an MPT converted from the Opticks GMaterial property map

* "Standardize" not a good name, its more "AdoptOpticksMaterialProperties"
  
   * BUT on the other hand it does standardize, because Opticks standardizes 
     materials to common wavelength domain when they are added to the GMaterialLib

* looks like this is not currently invoked by OKX4Test, only G4Opticks::TranslateGeometry

* Convertion with::

   G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ;


For legacy GDML this has some issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 2 extra OK materials (GlassSchottF2, MainH2OHale)  : the test glass comes after Air in the middle 
2. g4 material names are prefixed /dd/Materials/GdDopedLS


**/


void X4MaterialLib::Standardize()
{
    G4MaterialTable* mtab = G4Material::GetMaterialTable();
    const GMaterialLib* mlib = GMaterialLib::GetInstance();
    X4MaterialLib::Standardize( mtab, mlib ) ; 
}

void X4MaterialLib::Standardize( G4MaterialTable* mtab, const GMaterialLib* mlib )
{
    X4MaterialLib xmlib(mtab, mlib) ;  
}


X4MaterialLib::X4MaterialLib(G4MaterialTable* mtab, const GMaterialLib* mlib)
    :
    m_mtab(mtab),
    m_mlib(mlib)
{
    init();
}


const char* X4MaterialLib::DD_MATERIALS_PREFIX = "/dd/Materials/" ; 

/**
X4MaterialLib::init
---------------------

* Geant4 material names with the prefix "/dd/Materials/" 
  such as "/dd/Materials/LiquidScintillator" are regarded 
  to match Opticks unprefixed names "LiquidScintillator".
 
* Currently no name changes are made to the Geant4 materials.

**/

void X4MaterialLib::init()
{
    unsigned num_materials = m_mlib->getNumMaterials();
    unsigned num_m4 = G4Material::GetNumberOfMaterials() ;  
    bool match = num_materials == num_m4 ; 

    if(!match)
    {
       LOG(fatal) 
           << " num_materials MISMATCH "
           << " G4Material::GetNumberOfMaterials " << num_m4
           << " m_mlib->getNumMaterials " << num_materials
           ;

       for(unsigned i=0 ; i < num_m4 ; i++)
       {
           G4Material* m4 = (*m_mtab)[i] ; 
           const std::string& m4_name = m4->GetName();  
           std::cout << "m4 " << std::setw(3) << i << " : " << m4_name << std::endl ; 
       }
       for(unsigned i=0 ; i < num_materials ; i++)
       {
           GMaterial*  pmap = m_mlib->getMaterial(i); 
           const char* pmap_name = pmap->getName(); 
           std::cout << "mt " << std::setw(3) << i << " : " << pmap_name << std::endl ; 
       }

    }
    assert( match ); 

    for(unsigned i=0 ; i < num_materials ; i++)
    {
        GMaterial*  pmap = m_mlib->getMaterial(i); 
        G4Material* m4 = (*m_mtab)[i] ; 
        assert( pmap && m4 );  

        const char* pmap_name = pmap->getName(); 
        const std::string& m4_name = m4->GetName();  

        bool has_prefix = strncmp( m4_name.c_str(), DD_MATERIALS_PREFIX, strlen(DD_MATERIALS_PREFIX) ) == 0 ; 
        const char* m4_name_base = has_prefix ? m4_name.c_str() + strlen(DD_MATERIALS_PREFIX) : m4_name.c_str() ; 
        bool name_match = strcmp( m4_name_base, pmap_name) == 0 ;

        LOG(info) 
             << std::setw(5) << i 
             << " ok pmap_name " << std::setw(30) << pmap_name
             << " g4 m4_name  " << std::setw(30) << m4_name
             << " g4 m4_name_base  " << std::setw(30) << m4_name_base
             << " has_prefix " << has_prefix 
             ;     

        if(!name_match)
            LOG(fatal) 
                << " MATERIAL NAME MISMATCH " 
                << std::setw(5) << i 
                << " ok pmap_name " << std::setw(30) << pmap_name
                << " g4 m4_name  " << std::setw(30) << m4_name
                << " g4 m4_name_base  " << std::setw(30) << m4_name_base
                << " has_prefix " << has_prefix 
                ;

        assert(name_match ); 
        if( m4->GetMaterialPropertiesTable() == NULL ) continue ; 

        G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ; 
        m4->SetMaterialPropertiesTable( mpt ) ; 
    }
}



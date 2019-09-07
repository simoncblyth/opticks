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



void X4MaterialLib::init()
{
    unsigned num_materials = m_mlib->getNumMaterials();
    unsigned num_m4 = G4Material::GetNumberOfMaterials() ;  
    bool match = num_materials == num_m4 ; 

    if(!match)
       LOG(fatal) 
           << " num_materials MISMATCH "
           << " G4Material::GetNumberOfMaterials " << num_m4
           << " m_mlib->getNumMaterials " << num_materials
           ;

    assert( match ); 

    for(unsigned i=0 ; i < num_materials ; i++)
    {
        GMaterial*  pmap = m_mlib->getMaterial(i); 
        G4Material* m4 = (*m_mtab)[i] ; 
        assert( pmap && m4 );  

        const char* pmap_name = pmap->getName(); 
        const std::string& m4_name = m4->GetName();  
        bool name_match = strcmp( m4_name.c_str(), pmap_name) == 0 ;

        LOG(info) 
             << std::setw(5) << i 
             << " okmat " << std::setw(30) << pmap_name
             << " g4mat " << std::setw(30) << m4_name
             ;     


        if(!name_match)
            LOG(fatal) 
                << " MATERIAL NAME MISMATCH " 
                << " index " << i 
                << " pmap_name " << pmap_name
                << " m4_name " << m4_name
                ;

        assert(name_match ); 
        if( m4->GetMaterialPropertiesTable() == NULL ) continue ; 

        G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ; 
        m4->SetMaterialPropertiesTable( mpt ) ; 
    }
}



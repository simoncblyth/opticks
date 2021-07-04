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

#include <cassert>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "X4MaterialTable.hh"
#include "X4Material.hh"

#include "GMaterial.hh"
#include "GPropertyMap.hh"
#include "GMaterialLib.hh"
#include "Opticks.hh"

#include "PLOG.hh"


const plog::Severity X4MaterialTable::LEVEL = PLOG::EnvLevel("X4MaterialTable", "DEBUG") ; 


void X4MaterialTable::CollectAllMaterials(std::vector<G4Material*>& all_materials) // static
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    G4MaterialTable* mtab = G4Material::GetMaterialTable();

    LOG(LEVEL) << ". G4 nmat " << nmat ;  

    for(unsigned i=0 ; i < nmat ; i++)
    {   
        G4Material* material = (*mtab)[i];
        all_materials.push_back(material); 
    }
    LOG(LEVEL) << "collected all_materials.size  " << all_materials.size() ; 
}


/*
G4Material* X4MaterialTable::Get(unsigned idx)
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    assert( idx < nmat );
    G4MaterialTable* mtab = G4Material::GetMaterialTable();
    G4Material* material = (*mtab)[idx];

    //assert( material->GetIndex() == idx );
    // when the material table has been sorted with CMaterialSort 
    // the indices no longer match the positions 

    return material ; 
}
*/


void X4MaterialTable::Convert(GMaterialLib* mlib, std::vector<G4Material*>& materials_with_efficiency, const std::vector<G4Material*>& input_materials )
{
    assert( mlib->getNumMaterials() == 0 ); 
    X4MaterialTable xmt(mlib, materials_with_efficiency, input_materials ) ; 
    assert( mlib == xmt.getMaterialLib() );
}

GMaterialLib* X4MaterialTable::getMaterialLib()
{
    return m_mlib ;
}

X4MaterialTable::X4MaterialTable(GMaterialLib* mlib, std::vector<G4Material*>& materials_with_efficiency, const std::vector<G4Material*>& input_materials )
    :
    m_input_materials(input_materials),
    m_mlib(mlib),
    m_materials_with_efficiency(materials_with_efficiency)
{
    init();
}


/**
X4MaterialTable::init
-----------------------

For all m_input_materials G4Materials apply X4Material::Convert creating GMaterial 
which are collected in standardized and raw forms into m_mlib.

G4Material which have an EFFICIENCY property are collected into m_material_with_efficiency vector.

**/

void X4MaterialTable::init()
{
    unsigned num_input_materials = m_input_materials.size() ;

    LOG(LEVEL) << ". G4 nmat " << num_input_materials ;  

    for(unsigned i=0 ; i < num_input_materials ; i++)
    {   
        G4Material* material = m_input_materials[i] ; 
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

        if( mpt == NULL )
        {
            LOG(error) << "PROCEEDING TO convert material with no mpt " << material->GetName() ; 
            // continue ;  
        }
        else
        {
            LOG(LEVEL) << " converting material with mpt " <<  material->GetName() ; 
        }

        //char mode_oldstandardized = 'S' ;
        char mode_g4interpolated = 'G' ;
        GMaterial* mat = X4Material::Convert( material, mode_g4interpolated );   
        if(mat->hasProperty("EFFICIENCY")) m_materials_with_efficiency.push_back(material); 
        m_mlib->add(mat) ;    

        char mode_asis_nm = 'A' ;
        GMaterial* rawmat = X4Material::Convert( material, mode_asis_nm );   
        m_mlib->addRaw(rawmat) ;

        char mode_asis_en = 'E' ;
        GMaterial* rawmat_en = X4Material::Convert( material, mode_asis_en );   
        GPropertyMap<double>* pmap_rawmat_en = dynamic_cast<GPropertyMap<double>*>(rawmat_en) ; 
        m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib


    }
}



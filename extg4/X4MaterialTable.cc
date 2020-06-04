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
#include "GMaterialLib.hh"
#include "Opticks.hh"

#include "PLOG.hh"


const plog::Severity X4MaterialTable::LEVEL = verbose ; 



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

void X4MaterialTable::Convert(GMaterialLib* mlib, std::vector<G4Material*>& material_with_efficiency)
{
    assert( mlib->getNumMaterials() == 0 ); 
    X4MaterialTable xmt(mlib, material_with_efficiency ) ; 
    assert( mlib == xmt.getMaterialLib() );
}

GMaterialLib* X4MaterialTable::getMaterialLib()
{
    return m_mlib ;
}

X4MaterialTable::X4MaterialTable(GMaterialLib* mlib, std::vector<G4Material*>& material_with_efficiency)
    :
    m_mtab(G4Material::GetMaterialTable()),
    m_mlib(mlib),
    m_material_with_efficiency(material_with_efficiency)
{
    init();
}


void X4MaterialTable::init()
{
    unsigned nmat = G4Material::GetNumberOfMaterials();

    LOG(LEVEL) << ". G4 nmat " << nmat ;  

    for(unsigned i=0 ; i < nmat ; i++)
    {   
        G4Material* material = Get(i) ; 
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


        GMaterial* mat = X4Material::Convert( material ); 
        if(mat->hasProperty("EFFICIENCY"))
        {
             m_material_with_efficiency.push_back(material); 
        }

        //assert( mat->getIndex() == i ); // this is not the lib, no danger of triggering a close

        m_mlib->add(mat) ;    // creates standardized material
        m_mlib->addRaw(mat) ; // stores as-is
    }
}




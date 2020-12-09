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

#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "SPath.hh"
#include "PLOG.hh"

#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"
#include "CMaterialTable.hh"

const plog::Severity CMaterialTable::LEVEL = PLOG::EnvLevel("CMaterialTable", "DEBUG" ) ; 

CMaterialTable::CMaterialTable()
{
    init();
}
void CMaterialTable::init()
{
    initNameIndex();
}

void CMaterialTable::initNameIndex()
{
    const G4MaterialTable* mtab = G4Material::GetMaterialTable();
    unsigned nmat = G4Material::GetNumberOfMaterials();

    LOG(LEVEL)
              << " numOfMaterials " << nmat
              ;

    for(unsigned i=0 ; i < nmat ; i++)
    {
        G4Material* material = (*mtab)[i];
        G4String name_ = material->GetName() ;
        const char* name = name_.c_str();
        const char* shortname =  SPath::Basename(name) ;  // remove any prefix eg /dd/materials/Water -> Water 

        pLOG(LEVEL,+1) 
            << " index " << std::setw(3) << i 
            << " name " << std::setw(30) << name
            << " shortname " << std::setw(30) << shortname
            ;

        m_name2index[shortname] = i ;   
        m_index2name[i] = shortname ;   
    }
}

void CMaterialTable::dump(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<unsigned, std::string> MUS ; 
    for(MUS::const_iterator it=m_index2name.begin() ; it != m_index2name.end() ; it++)
        std::cout 
             << std::setw(35) << it->first 
             << std::setw(25) << it->second
             << std::endl ; 

}

void CMaterialTable::fillMaterialIndexMap( std::map<std::string, unsigned int>&  mixm )
{
    typedef std::map<std::string, unsigned> MSU ; 
    for(MSU::const_iterator it=m_name2index.begin() ; it != m_name2index.end() ; it++)
    {
         std::string name = it->first ; 
         unsigned index = it->second ; 
         mixm[name] = index ;  
    }
}

const std::map<std::string, unsigned>& CMaterialTable::getMaterialMap() const 
{
   return m_name2index ;  
}


unsigned CMaterialTable::getMaterialIndex(const char* shortname)
{
    return m_name2index.count(shortname) == 1 ? m_name2index[shortname] : -1 ; 
}

void CMaterialTable::dumpMaterial(const char* shortname)
{
     unsigned index = getMaterialIndex(shortname);
     dumpMaterial(index);
}


void CMaterialTable::dumpMaterial(unsigned index)
{
    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    unsigned numOfMaterials = G4Material::GetNumberOfMaterials();

    G4Material* material = index < numOfMaterials ? (*theMaterialTable)[index] : NULL ;
    dumpMaterial(material);
}


void CMaterialTable::dumpMaterial(G4Material* material)
{
    if(!material) return ; 
    G4String name = material->GetName() ;

    CMPT mpt(material->GetMaterialPropertiesTable());

    LOG(info) << "CMaterialTable::dumpMaterial "
              << name 
              ;


    mpt.dump("MPT:");

    mpt.dumpProperty("RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB");

}





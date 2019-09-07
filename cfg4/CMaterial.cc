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

#include "SDigest.hh"

#include "CMPT.hh"
#include "CMaterial.hh"

std::string CMaterial::Digest(G4Material* material)
{
    if(!material) return "" ; 
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable() ; 
    const G4String& name = material->GetName();      
    std::string dmpt = CMPT::Digest(mpt) ; 
    SDigest dig ;
    dig.update( const_cast<char*>(name.data()), name.size() );  
    dig.update( const_cast<char*>(dmpt.data()), dmpt.size() );  
    return dig.finalize();
}
    
CMaterial::CMaterial(G4Material* mat) 
   :
   m_material(mat)
{
}

G4Material* CMaterial::getMaterial() const 
{
    return m_material ; 
}

std::string CMaterial::digest() const 
{
    return Digest(m_material); 
}



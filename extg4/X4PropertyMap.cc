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


#include "G4MaterialPropertyVector.hh"
#include "G4MaterialPropertiesTable.hh"

#include "X4PropertyMap.hh"
#include "X4Property.hh"

#include "GPropertyMap.hh"
#include "GProperty.hh"
#include "GMaterialLib.hh"


G4MaterialPropertiesTable* X4PropertyMap::Convert( const GPropertyMap<double>* pmap )
{  
    X4PropertyMap xpm(pmap); 
    return xpm.getMPT(); 
}

X4PropertyMap::X4PropertyMap(const GPropertyMap<double>* pmap) 
    :
    m_pmap(pmap),
    m_mpt(new G4MaterialPropertiesTable),
    m_mlib(GMaterialLib::Get())
{
    init();
}

G4MaterialPropertiesTable* X4PropertyMap::getMPT() const 
{
    return m_mpt ; 
}

void X4PropertyMap::init()
{
    unsigned num_prop = m_pmap->getNumProperties();    
    for(unsigned i=0 ; i<num_prop ; i++) 
    {
         const char* key =  m_pmap->getPropertyNameByIndex(i);  // refractive_index absorption_length scattering_length reemission_prob
         const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
         GProperty<double>* prop = m_pmap->getPropertyByIndex(i);

         G4PhysicsVector* pvec = X4Property<double>::Convert( prop ) ; 

         G4MaterialPropertyVector* mpv =  dynamic_cast<G4MaterialPropertyVector*>(pvec); 

         m_mpt->AddProperty( lkey, mpv ); 
    }
}







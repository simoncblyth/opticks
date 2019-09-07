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

#include "BStr.hh"
#include <iomanip>

#include "G4OpticalSurface.hh"
#include "GSurfaceLib.hh"

#include "CSurfaceBridge.hh"
#include "CSkinSurfaceTable.hh"
#include "CBorderSurfaceTable.hh"

#include "PLOG.hh"

CSurfaceBridge::CSurfaceBridge( GSurfaceLib* slib) 
    :
    m_slib(slib),
    m_skin(new CSkinSurfaceTable),
    m_border(new CBorderSurfaceTable)
{
    initMap(m_skin);
    initMap(m_border);
}

void CSurfaceBridge::initMap(CSurfaceTable* stab)
{
    LOG(verbose) << "CSurfaceBridge::initMap" 
               << " stab " << stab->getName()
              ;

    unsigned nsurf = stab->getNumSurf();

    for(unsigned i=0 ; i < nsurf ; i++)
    {
        const G4OpticalSurface* os = stab->getSurface(i);

        std::string name = os->GetName() ;

        const char* shortname = BStr::afterLastOrAll( name.c_str(), '/' );

        unsigned index =  m_slib->getIndex( shortname );

        m_g4toix[os] = index ; 

        m_ixtoname[index] = shortname ;


        LOG(verbose) << " i " << std::setw(3) << i 
                  << " name " << std::setw(35) << name 
                  << " shortname " << std::setw(35) << shortname 
                  << " index " << std::setw(5)  << index
                  ; 
    }
}



void CSurfaceBridge::dumpMap(const char* msg)
{
    LOG(info) << msg << " g4toix.size " << m_g4toix.size() ;

    typedef std::map<const G4OpticalSurface*, unsigned> SU ; 
    for(SU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
         const G4OpticalSurface* surf = it->first ; 
         unsigned index = it->second ; 

         std::cout << std::setw(50) << surf->GetName() 
                   << std::setw(10) << index 
                   << std::endl ; 

         unsigned check = getSurfaceIndex(surf);
         assert(check == index);
    }
}


void CSurfaceBridge::dump(const char* msg)
{
    LOG(info) << msg << " g4toix.size " << m_g4toix.size() ;

    m_skin->dump(msg);
    m_border->dump(msg);


    typedef std::vector<const G4OpticalSurface*> S ; 
    S surfs ; 

    typedef std::map<const G4OpticalSurface*, unsigned> SU ; 
    for(SU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++) 
         surfs.push_back(it->first);

    std::stable_sort( surfs.begin(), surfs.end(), *this );          

    for(S::const_iterator it=surfs.begin() ; it != surfs.end() ; it++)
    {
        const G4OpticalSurface* surf = *it ;  
        unsigned index = getSurfaceIndex(surf);
        const char* shortname = getSurfaceName(index);

        std::cout << std::setw(50) << surf->GetName() 
                  << std::setw(10) << index 
                  << std::setw(30) << shortname 
                  << std::endl ; 
    }
}


bool CSurfaceBridge::operator()(const G4OpticalSurface* a, const G4OpticalSurface* b)
{
    unsigned ia = getSurfaceIndex(a);
    unsigned ib = getSurfaceIndex(b);
    return ia < ib ; 
}



unsigned int CSurfaceBridge::getSurfaceIndex(const G4OpticalSurface* surf)
{
    return m_g4toix[surf] ;
}
const char* CSurfaceBridge::getSurfaceName(unsigned int index)
{
    return m_ixtoname[index].c_str() ;
}


const G4OpticalSurface* CSurfaceBridge::getG4Surface(unsigned int qindex) // 0-based Opticks surface index to G4OpticalSurface
{
    typedef std::map<const G4OpticalSurface*, unsigned> SU ; 
    const G4OpticalSurface* surf = NULL ; 
    for(SU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
         unsigned index = it->second ; 
         if(index == qindex)
         {
             surf = it->first ; 
             break ;
         }
    }
    return surf ; 
}



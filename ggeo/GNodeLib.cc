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

#include <sstream>
#include <iostream>
#include <iomanip>

#include "PLOG.hh"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GVolume.hh"
#include "GNodeLib.hh"
#include "GTreePresent.hh"


const char* GNodeLib::GetRelDir(bool analytic, bool test)
{
    std::stringstream ss ; 
    ss << "GNodeLib" ; 
    if(analytic) ss << "Analytic" ; 
    if(test)     ss << "Test" ; 

    std::string s = ss.str() ;
    return strdup(s.c_str()) ;
}

GNodeLib* GNodeLib::Load(Opticks* ok, bool analytic, bool test)
{
    GNodeLib* nodelib = new GNodeLib(ok, analytic, test) ;
    nodelib->loadFromCache();
    return nodelib ; 
}

void GNodeLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;
    m_pvlist = GItemList::Load(idpath, "PVNames", m_reldir);
    m_lvlist = GItemList::Load(idpath, "LVNames", m_reldir);
}

GNodeLib::GNodeLib(Opticks* ok, bool analytic, bool test, GNodeLib* basis)  
    :
    m_ok(ok),
    m_analytic(analytic),
    m_test(test),
    m_basis(basis),
    m_reldir(GetRelDir(analytic,test)),
    m_pvlist(NULL),
    m_lvlist(NULL),
    m_treepresent(new GTreePresent(100, 1000))   // depth_max,sibling_max
{
}

GNodeLib* GNodeLib::getBasis() const 
{
    return m_basis ; 
}


void GNodeLib::save() const 
{
    const char* idpath = m_ok->getIdPath() ;
    LOG(debug) << "GNodeLib::save"
              << " idpath " << idpath 
              ;

    if(m_pvlist)
    {
        m_pvlist->save(idpath);
    }
    else
    {
        LOG(warning) << "GNodeLib::save pvlist NULL " ; 
    }


    if(m_lvlist)
    {
        m_lvlist->save(idpath);
    }
    else
    {
        LOG(warning) << "GNodeLib::save lvlist NULL " ; 
    }


    GNode* top = getNode(0); 
    m_treepresent->traverse(top);
    m_treepresent->write(idpath, m_reldir);
}


std::string GNodeLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GNodeLib"
       << " reldir " << ( m_reldir ? m_reldir : "-" )
       << " numPV " << getNumPV()
       << " numLV " << getNumLV()
       << " numVolumes " << getNumVolumes()
       << " PV(0) " << getPVName(0)
       << " LV(0) " << getLVName(0)
       ;


    typedef std::map<unsigned, GVolume*>::const_iterator IT ; 

    IT beg = m_volumemap.begin() ;
    IT end = m_volumemap.end() ;

    for(IT it=beg ; it != end && std::distance(beg,it) < 10 ; it++)
    {
        ss << " ( " << it->first << " )" ; 
    }

    return ss.str();
}



unsigned GNodeLib::getNumPV() const 
{
    unsigned npv = m_pvlist->getNumKeys(); 
    return npv ; 
}

unsigned GNodeLib::getNumLV() const 
{
    unsigned nlv = m_lvlist->getNumKeys(); 
    return nlv ; 
}

const char* GNodeLib::getPVName(unsigned int index) const 
{
    return m_pvlist ? m_pvlist->getKey(index) : NULL ; 
}
const char* GNodeLib::getLVName(unsigned int index) const 
{
    return m_lvlist ? m_lvlist->getKey(index) : NULL ; 
}


unsigned GNodeLib::getNumVolumes() const 
{
    return m_volumes.size();
}
std::vector<GVolume*>& GNodeLib::getVolumes() 
{
    return m_volumes ; 
}


GItemList* GNodeLib::getPVList()
{
    return m_pvlist ; 
}
GItemList* GNodeLib::getLVList()
{
    return m_lvlist ; 
}



void GNodeLib::add(GVolume* volume)
{
    m_volumes.push_back(volume);

    unsigned int index = volume->getIndex(); 

    if(m_test)
    {
        assert( m_volumes.size() - 1 == index && "indices of test geometry volumes added to GNodeLib must follow the sequence : 0,1,2,... " );
    }


    //assert( m_volumemap.size() == index );   //  only with relative GVolume indexing

/*
    LOG(info) << "GNodeLib::add"
              << " volumeIndex " << index 
              << " preCount " << m_volumemap.size()
              ;
*/
              
    m_volumemap[index] = volume ; 

    if(!m_pvlist) m_pvlist = new GItemList("PVNames", m_reldir) ; 
    if(!m_lvlist) m_lvlist = new GItemList("LVNames", m_reldir) ; 

    m_lvlist->add(volume->getLVName()); 
    m_pvlist->add(volume->getPVName()); 

    // NB added in tandem, so same counts and same index as the volumes  

    GVolume* check = getVolume(index);
    assert(check == volume);
}




unsigned GNodeLib::addSensorVolume(GVolume* volume)
{
    unsigned sensorIndex = m_sensor_volumes.size() ;  
    m_sensor_volumes.push_back(volume); 
    return sensorIndex ; 
}
unsigned GNodeLib::getNumSensorVolumes() const 
{
    return m_sensor_volumes.size(); 
}
GVolume* GNodeLib::getSensorVolume(unsigned sensorIndex) const 
{
    return m_sensor_volumes[sensorIndex]; 
}






GVolume* GNodeLib::getVolume(unsigned index) const 
{
    GVolume* volume = NULL ; 
    if(m_volumemap.find(index) != m_volumemap.end()) 
    {
        volume = m_volumemap.at(index) ;
        assert(volume->getIndex() == index);
    }
    return volume ; 
}

GVolume* GNodeLib::getVolumeSimple(unsigned int index)
{
    return m_volumes[index];
}


GNode* GNodeLib::getNode(unsigned index) const 
{
    GVolume* volume = getVolume(index);
    GNode* node = static_cast<GNode*>(volume); 
    return node ; 
}


void GNodeLib::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    LOG(info) << " NumVolumes " << m_volumes.size() ; 

    for(unsigned i=0 ; i < std::min(m_volumes.size(), 100ul) ; i++ )
    {
        GVolume* volume = m_volumes.at(i) ; 
        std::cout 
            << " ix " << std::setw(5) << i 
            << " lv " << std::setw(40) << volume->getLVName()
            << " pv " << std::setw(40) << volume->getPVName()
            << std::endl 
            ;
    }
}




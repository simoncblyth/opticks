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
#include "NPY.hpp"
#include "NBBox.hpp"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GVolume.hh"
#include "GNodeLib.hh"
#include "GTreePresent.hh"

const plog::Severity GNodeLib::LEVEL = PLOG::EnvLevel("GNodeLib", "INFO"); 

const char* GNodeLib::RELDIR = "GNodeLib" ; 

const char* GNodeLib::CacheDir(const Opticks* ok)
{
    std::string cachedir = ok->getObjectPath(RELDIR) ; 
    return strdup(cachedir.c_str()); 
}

GNodeLib::GNodeLib(Opticks* ok)  
    :
    m_ok(ok),
    m_loading(false),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(new GItemList("PVNames", m_reldir)),
    m_lvlist(new GItemList("LVNames", m_reldir)),
    m_transforms(NPY<float>::make(0,4,4)),
    m_bounding_box(NPY<float>::make(0,2,4)),
    m_center_extent(NPY<float>::make(0,4)),
    m_treepresent(new GTreePresent(100, 1000))   // depth_max,sibling_max
{
}

GNodeLib::GNodeLib(Opticks* ok, bool loading)
    :
    m_ok(ok),
    m_loading(loading),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(GItemList::Load(ok->getIdPath(), "PVNames",  m_reldir)),
    m_lvlist(GItemList::Load(ok->getIdPath(), "LVNames",  m_reldir)),
    m_transforms(NPY<float>::load(m_cachedir,"transforms.npy")),
    m_bounding_box(NPY<float>::load(m_cachedir,"bounding_box.npy")),
    m_center_extent(NPY<float>::load(m_cachedir,"center_extent.npy")),
    m_treepresent(NULL)
{
}
 
GNodeLib* GNodeLib::Load(Opticks* ok)
{
    bool loading = true ; 
    return new GNodeLib(ok, loading); 
}

void GNodeLib::save() const 
{
    const char* idpath = m_ok->getIdPath() ;
    LOG(LEVEL) << " idpath " << idpath ; 
    m_pvlist->save(idpath);
    m_lvlist->save(idpath);

    m_transforms->save(m_cachedir, "transforms.npy"); 
    m_bounding_box->save(m_cachedir, "bounding_box.npy"); 
    m_center_extent->save(m_cachedir, "center_extent.npy"); 

    const GNode* top = getNode(0); 
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

    typedef std::map<unsigned, const GVolume*>::const_iterator IT ; 

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
std::vector<const GVolume*>& GNodeLib::getVolumes() 
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


/**
GNodeLib::add
---------------

Collects all volume information 


**/

void GNodeLib::add(const GVolume* volume)
{
    m_volumes.push_back(volume);
    unsigned int index = volume->getIndex(); 
    assert( m_volumes.size() - 1 == index && "indices of test geometry volumes added to GNodeLib must follow the sequence : 0,1,2,... " ); // formerly only for m_test
    m_volumemap[index] = volume ; 

    glm::mat4 transform = volume->getTransformMat4();
    m_transforms->add(transform);  

    nbbox* bb = volume->getVerticesBBox(); 
    glm::vec4 min(bb->min, 1.f); 
    glm::vec4 max(bb->max, 1.f); 
    m_bounding_box->add( min, max); 

    glm::vec4 ce = bb->ce(); 
    m_center_extent->add(ce); 

    m_lvlist->add(volume->getLVName()); 
    m_pvlist->add(volume->getPVName()); 
    // NB added in tandem, so same counts and same index as the volumes  

    const GVolume* check = getVolume(index);
    assert(check == volume);
}



unsigned GNodeLib::addSensorVolume(const GVolume* volume)
{
    unsigned sensorIndex = m_sensor_volumes.size() ;  
    m_sensor_volumes.push_back(volume); 
    return sensorIndex ; 
}
unsigned GNodeLib::getNumSensorVolumes() const 
{
    return m_sensor_volumes.size(); 
}
const GVolume* GNodeLib::getSensorVolume(unsigned sensorIndex) const 
{
    return m_sensor_volumes[sensorIndex]; 
}

std::string GNodeLib::reportSensorVolumes(const char* msg) const 
{
    std::stringstream ss ; 
    unsigned numSensorVolumes = getNumSensorVolumes(); 
    ss
        << msg 
        << " numSensorVolumes " << numSensorVolumes
        ;
    return ss.str(); 
}

void GNodeLib::dumpSensorVolumes(const char* msg) const 
{
    LOG(info) << reportSensorVolumes(msg) ; 
    unsigned numSensorVolumes = getNumSensorVolumes(); 
    for(unsigned i=0 ; i < numSensorVolumes ; i++)
    {
        unsigned sensorIdx = i ; 
        const GVolume* sensor = getSensorVolume(sensorIdx) ; 
        assert(sensor); 
        const char* sensorPVName =  sensor->getPVName() ; 
        assert(sensorPVName);
        const void* const sensorOrigin = sensor->getOriginNode() ;
        assert(sensorOrigin);  
        const GVolume* outer = sensor->getOuterVolume() ; 
        assert(outer);
        const char* outerPVName = outer->getPVName() ;
        assert(outerPVName);
        const void* const outerOrigin = outer->getOriginNode() ;  
        assert(outerOrigin);

        std::cout 
            << " sensorIdx " << std::setw(6) << sensorIdx
            << " sensor " << std::setw(8) << sensor 
            << " outer " << std::setw(8) << outer 
            << " sensorPVName " << std::setw(50) << sensorPVName
            << " outerPVName "  << std::setw(50) << outerPVName
            << " sensorOrigin " << std::setw(8) << sensorOrigin
            << " outerOrigin " << std::setw(8) << outerOrigin
            << std::endl 
            ;
    }
}

void GNodeLib::getSensorPlacements(std::vector<void*>& placements) const 
{
    unsigned numSensorVolumes = getNumSensorVolumes(); 
    for(unsigned i=0 ; i < numSensorVolumes ; i++)
    {
        unsigned sensorIdx = i ; 
        const GVolume* sensor = getSensorVolume(sensorIdx) ; 
        assert(sensor); 
        const GVolume* outer = sensor->getOuterVolume() ; 
        assert(outer); 
        void* outerOrigin = outer->getOriginNode() ;  
        placements.push_back(outerOrigin); 
    }
}


const GVolume* GNodeLib::getVolume(unsigned index) const 
{
    const GVolume* volume = NULL ; 
    if(m_volumemap.find(index) != m_volumemap.end()) 
    {
        volume = m_volumemap.at(index) ;
        assert(volume->getIndex() == index);
    }
    return volume ; 
}


GVolume* GNodeLib::getVolumeNonConst(unsigned index)
{
    const GVolume* volume = getVolume(index); 
    return const_cast<GVolume*>(volume); 
}

const GVolume* GNodeLib::getVolumeSimple(unsigned int index)
{
    return m_volumes[index];
}


const GNode* GNodeLib::getNode(unsigned index) const 
{
    const GVolume* volume = getVolume(index);
    const GNode* node = static_cast<const GNode*>(volume); 
    return node ; 
}


void GNodeLib::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    LOG(info) << " NumVolumes " << m_volumes.size() ; 

    for(unsigned i=0 ; i < std::min(m_volumes.size(), 100ul) ; i++ )
    {
        const GVolume* volume = m_volumes.at(i) ; 
        std::cout 
            << " ix " << std::setw(5) << i 
            << " lv " << std::setw(40) << volume->getLVName()
            << " pv " << std::setw(40) << volume->getPVName()
            << std::endl 
            ;
    }
}




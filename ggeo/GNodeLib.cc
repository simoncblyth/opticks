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
#include "GLMFormat.hpp"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GVolume.hh"
#include "GNodeLib.hh"
#include "GTreePresent.hh"

const plog::Severity GNodeLib::LEVEL = PLOG::EnvLevel("GNodeLib", "INFO"); 

const char* GNodeLib::RELDIR = "GNodeLib" ; 
const char* GNodeLib::PV = "all_volume_PVNames" ; 
const char* GNodeLib::LV = "all_volume_LVNames" ; 
const char* GNodeLib::TR = "all_volume_transforms.npy" ; 
const char* GNodeLib::CE = "all_volume_center_extent.npy" ; 
const char* GNodeLib::BB = "all_volume_bbox.npy" ; 
const char* GNodeLib::ID = "all_volume_identity.npy" ; 
const char* GNodeLib::NI = "all_volume_nodeinfo.npy" ; 

const char* GNodeLib::CacheDir(const Opticks* ok)  // static
{
    std::string cachedir = ok->getObjectPath(RELDIR) ; 
    return strdup(cachedir.c_str()); 
}

GNodeLib* GNodeLib::Load(Opticks* ok)  // static
{
    bool loading = true ; 
    return new GNodeLib(ok, loading); 
}

GNodeLib::GNodeLib(Opticks* ok)  
    :
    m_ok(ok),
    m_keydir(ok->getIdPath()),
    m_loading(false),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(new GItemList(PV, m_reldir)),
    m_lvlist(new GItemList(LV, m_reldir)),
    m_transforms(NPY<float>::make(0,4,4)),
    m_bounding_box(NPY<float>::make(0,2,4)),
    m_center_extent(NPY<float>::make(0,4)),
    m_identity(NPY<unsigned>::make(0,4)),
    m_nodeinfo(NPY<unsigned>::make(0,4)),
    m_treepresent(new GTreePresent(100, 1000)),   // depth_max,sibling_max
    m_num_volumes(0),
    m_volumes(0)
{
    LOG(LEVEL) << "created" ; 
}

GNodeLib::GNodeLib(Opticks* ok, bool loading)
    :
    m_ok(ok),
    m_keydir(ok->getIdPath()),
    m_loading(loading),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(GItemList::Load(ok->getIdPath(), PV, m_reldir)),
    m_lvlist(GItemList::Load(ok->getIdPath(), LV, m_reldir)),
    m_transforms(NPY<float>::load(m_cachedir, TR)),
    m_bounding_box(NPY<float>::load(m_cachedir, BB)),
    m_center_extent(NPY<float>::load(m_cachedir,CE)),
    m_identity(NPY<unsigned>::load(m_cachedir,ID)),
    m_nodeinfo(NPY<unsigned>::load(m_cachedir,NI)),
    m_treepresent(NULL),
    m_num_volumes(initNumVolumes()),
    m_volumes(0)
{
    LOG(LEVEL) << "loaded" ; 
}
 
unsigned GNodeLib::initNumVolumes() const
{
    unsigned num_volumes = m_pvlist->getNumKeys(); 
    assert( m_lvlist->getNumKeys() == num_volumes ); 
    assert( m_transforms->getNumItems() == num_volumes ); 
    assert( m_bounding_box->getNumItems() == num_volumes ); 
    assert( m_center_extent->getNumItems() == num_volumes ); 
    assert( m_identity->getNumItems() == num_volumes ); 
    assert( m_nodeinfo->getNumItems() == num_volumes ); 

/*
    huh giving crazy values
    unsigned nvol = m_volumes.size() ;
    if( nvol != 0 )
       LOG(fatal) 
          << " unexpected "
          << " num_volumes " << num_volumes
          << " nvol " << nvol
          ;
              
    assert( nvol == 0 );  // zero live volumes  postcache
*/


    return num_volumes ; 
}

unsigned GNodeLib::getNumVolumes() const 
{
    return m_num_volumes ; 
}


void GNodeLib::save() const 
{
    LOG(LEVEL) << " keydir " << m_keydir ; 
    m_pvlist->save(m_keydir);
    m_lvlist->save(m_keydir);

    m_transforms->save(m_cachedir,  TR); 
    m_bounding_box->save(m_cachedir, BB); 
    m_center_extent->save(m_cachedir, CE); 
    m_identity->save(m_cachedir, ID); 
    m_nodeinfo->save(m_cachedir, NI); 

    if(m_treepresent)  // pre-cache only as needs the full node tree
    {
        const GNode* top = getNode(0); 
        m_treepresent->traverse(top);
        m_treepresent->write(m_keydir, m_reldir);
    }
}

std::string GNodeLib::getShapeString() const 
{
    std::stringstream ss ; 
    ss 
       << std::endl << std::setw(20) << TR << " " << m_transforms->getShapeString() 
       << std::endl << std::setw(20) << BB << " " << m_bounding_box->getShapeString() 
       << std::endl << std::setw(20) << CE << " " << m_center_extent->getShapeString() 
       << std::endl << std::setw(20) << ID << " " << m_identity->getShapeString() 
       << std::endl << std::setw(20) << NI << " " << m_nodeinfo->getShapeString() 
       ;
    return ss.str();
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

const char* GNodeLib::getPVName(unsigned index) const 
{
    return m_pvlist ? m_pvlist->getKey(index) : NULL ; 
}
const char* GNodeLib::getLVName(unsigned index) const 
{
    return m_lvlist ? m_lvlist->getKey(index) : NULL ; 
}

/**
GNodeLib::getVolumes
----------------------

Returns empty vector postcache.

**/

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
    unsigned index = volume->getIndex(); 
    m_volumes.push_back(volume);
    assert( m_volumes.size() - 1 == index && "indices of the geometry volumes added to GNodeLib must follow the sequence : 0,1,2,... " ); // formerly only for m_test
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

    glm::uvec4 id = volume->getIdentity_(); 
    m_identity->add(id);

    glm::uvec4 ni = volume->getNodeInfo_(); 
    m_nodeinfo->add(ni);

    const GVolume* check = getVolume(index);
    assert(check == volume);

    m_num_volumes += 1 ; 
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


/**
GNodeLib::getVolume  (precache)
--------------------------------
**/

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

/**
GNodeLib::getVolumeNonConst (precache)
-----------------------------------------
**/

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






unsigned GNodeLib::getNumTransforms() const 
{
    unsigned num_transforms = m_transforms->getNumItems(); 
    return num_transforms ; 
}
glm::mat4 GNodeLib::getTransform(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::mat4 tr = m_transforms->getMat4(index) ; 
    return tr ;  
}
glm::vec4 GNodeLib::getCE(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::vec4 ce = m_center_extent->getQuad(index) ; 
    return ce ;  
}
glm::uvec4 GNodeLib::getIdentity(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::uvec4 id = m_identity->getQuad(index) ; 
    return id ;  
}
glm::uvec4 GNodeLib::getNodeInfo(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::uvec4 ni = m_nodeinfo->getQuad(index) ; 
    return ni ;  
}



void GNodeLib::getBB(unsigned index, glm::vec4& mn, glm::vec4& mx ) const 
{
    assert( index < m_num_volumes ); 
    mn = m_bounding_box->getQuad(index, 0); 
    mx = m_bounding_box->getQuad(index, 1); 
}

nbbox GNodeLib::getBBox(unsigned index) const 
{
    glm::vec4 mn ; 
    glm::vec4 mx ;
    getBB(index, mn, mx); 
    return make_bbox( mn.x, mn.y, mn.z, mx.x, mx.y, mx.z );  
}




/**
GNodeLib::dump  (precache)
----------------------------

Precache dumper.

**/
void GNodeLib::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    LOG(info) << " m_volumes.size() " << m_volumes.size() ; 

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

void GNodeLib::Dump(const char* msg) const 
{
    LOG(info) << msg ; 
    std::cout << desc() << std::endl ; 
    std::cout << getShapeString() << std::endl ; 
    dump(msg); 
}


std::string GNodeLib::descVolume(unsigned index) const
{
    bool test = m_ok->isTest() ;    // --test : dumping volumes
    glm::vec4 ce = getCE(index);
    const char* lvn = test ? "test" : getLVName(index) ; 

    std::stringstream ss ; 
    ss
       << " " 
       << std::setw(7) << index 
       << std::setw(70) << lvn 
       << " "  
       << gpresent( "ce ", ce )
       ;  

    return ss.str(); 
}


void GNodeLib::dumpVolumes(const char* msg, float extent_cut_mm, int cursor ) const 
{
    unsigned num_volumes = getNumVolumes();
    LOG(info) << msg  << " num_volumes " << num_volumes ;    

    LOG(info) << "first volumes "  ; 
    for(unsigned i=0 ; i < std::min(num_volumes, 20u) ; i++) 
    {    
        std::cout 
            << ( int(i) == cursor ? " **" : "   " ) 
            << descVolume(i)
            ;
    }    

    LOG(info) << "volumes with extent greater than " << extent_cut_mm << " mm " ; 
    for(unsigned i=0 ; i < num_volumes ; i++) 
    {    
        glm::vec4 ce = getCE(i);
        if(ce.w > extent_cut_mm )
        std::cout 
            << ( int(i) == cursor ? " **" : "   " ) 
            << descVolume(i)
            ;    
    }    
}


/**
GNodeLib::findContainerVolumeIndex
-----------------------------------

Returns the absolute volume index of the smallest volume 
that contains the provided coordinate.

NB simple slow implementation as this has only been used 
interactively with unprojected frame positions, see Scene::touch.

Contrast with GMergedMesh::findContainer (actually GMesh::findContainer but it only makes
sense in the subclass) which does the same within the volumes of that mesh only 
and returns a volume index local to that merged mesh.

**/

unsigned GNodeLib::findContainerVolumeIndex(float x, float y, float z) const 
{
    unsigned container(0);
    float cext(FLT_MAX) ; 

    for(unsigned index=0 ; index < m_num_volumes ; index++)
    {    
         glm::vec4 ce = getCE(index) ;
         glm::vec3 hi(ce.x + ce.w, ce.y + ce.w, ce.z + ce.w );
         glm::vec3 lo(ce.x - ce.w, ce.y - ce.w, ce.z - ce.w );

         if(  
              x > lo.x && x < hi.x  &&
              y > lo.y && y < hi.y  &&
              z > lo.z && z < hi.z 
           )    
          {    
               if(ce.w < cext)
               {    
                   cext = ce.w ; 
                   container = index ; 
               }    
          }    
    }    
    return container ; 
}





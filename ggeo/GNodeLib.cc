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
#include "OpticksIdentity.hh"

#include "GItemList.hh"
#include "GVolume.hh"
#include "GNodeLib.hh"
#include "GTreePresent.hh"

const plog::Severity GNodeLib::LEVEL = PLOG::EnvLevel("GNodeLib", "DEBUG"); 

const char* GNodeLib::RELDIR = "GNodeLib" ; 
const char* GNodeLib::PV = "all_volume_PVNames" ; 
const char* GNodeLib::LV = "all_volume_LVNames" ; 
const char* GNodeLib::TR = "all_volume_transforms.npy" ; 
const char* GNodeLib::IT = "all_volume_inverse_transforms.npy" ; 
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
    m_loaded(false),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(new GItemList(PV, m_reldir)),
    m_lvlist(new GItemList(LV, m_reldir)),
    m_transforms(NPY<float>::make(0,4,4)),
    m_inverse_transforms(NPY<float>::make(0,4,4)),
    m_bounding_box(NPY<float>::make(0,2,4)),
    m_center_extent(NPY<float>::make(0,4)),
    m_identity(NPY<unsigned>::make(0,4)),
    m_nodeinfo(NPY<unsigned>::make(0,4)),
    m_treepresent(new GTreePresent(100, 1000)),   // depth_max,sibling_max
    m_num_volumes(0),
    m_num_sensors(0),
    m_volumes(0),
    m_root(NULL)
{
    LOG(LEVEL) << "created" ; 
    assert( m_loaded == false ); 
}

GNodeLib::GNodeLib(Opticks* ok, bool loading)
    :
    m_ok(ok),
    m_keydir(ok->getIdPath()),
    m_loaded(loading),
    m_cachedir(GNodeLib::CacheDir(ok)),
    m_reldir(RELDIR),
    m_pvlist(GItemList::Load(ok->getIdPath(), PV, m_reldir)),
    m_lvlist(GItemList::Load(ok->getIdPath(), LV, m_reldir)),
    m_transforms(NPY<float>::load(m_cachedir, TR)),
    m_inverse_transforms(NPY<float>::load(m_cachedir, IT)),
    m_bounding_box(NPY<float>::load(m_cachedir, BB)),
    m_center_extent(NPY<float>::load(m_cachedir,CE)),
    m_identity(NPY<unsigned>::load(m_cachedir,ID)),
    m_nodeinfo(NPY<unsigned>::load(m_cachedir,NI)),
    m_treepresent(NULL),
    m_num_volumes(initNumVolumes()),
    m_num_sensors(initSensorIdentity()),
    m_volumes(0),
    m_root(NULL)
{
    LOG(LEVEL) << "loaded" ; 
    assert( m_loaded == true ); 
    assert( m_sensor_identity.size() == m_num_sensors ); 
}

 
unsigned GNodeLib::initNumVolumes() const
{
    assert( m_loaded ); 
    unsigned num_volumes = m_pvlist->getNumKeys(); 
    assert( m_lvlist->getNumKeys() == num_volumes ); 
    assert( m_transforms->getNumItems() == num_volumes ); 
    assert( m_inverse_transforms->getNumItems() == num_volumes ); 
    assert( m_bounding_box->getNumItems() == num_volumes ); 
    assert( m_center_extent->getNumItems() == num_volumes ); 
    assert( m_identity->getNumItems() == num_volumes ); 
    assert( m_nodeinfo->getNumItems() == num_volumes ); 
    return num_volumes ; 
}

/**
GNodeLib::initSensorIdentity
-----------------------------

Loops over m_identity volume identity array, collecting 
identity quads for volumes with a sensorIndex assigned
into m_sensor_identity.  Returns the number of such sensor volumes. 

**/

unsigned GNodeLib::initSensorIdentity() 
{
    assert( m_loaded ); 
    for(unsigned i=0 ; i < m_num_volumes ; i++)
    {
        glm::uvec4 id = m_identity->getQuad_(i); 
        unsigned sensorIndex = id.w ; 
        if( sensorIndex == GVolume::SENSOR_UNSET ) continue ; 
        //LOG(info) << " id " << glm::to_string(id) << " sensorIndex " << sensorIndex  ; 
        m_sensor_identity.push_back(id); 
    } 
    unsigned num_sensors = m_sensor_identity.size() ;
    LOG(LEVEL) 
        << " m_num_volumes " << m_num_volumes 
        << " num_sensors " << num_sensors 
        ;
    return num_sensors ; 
}

unsigned GNodeLib::getNumVolumes() const 
{
    return m_num_volumes ; 
}

Opticks* GNodeLib::getOpticks() const 
{
    return m_ok ; 
}

void GNodeLib::save() const 
{
    LOG(LEVEL) << " keydir " << m_keydir ; 
    m_pvlist->save(m_keydir);
    m_lvlist->save(m_keydir);

    m_transforms->save(m_cachedir,  TR); 
    m_inverse_transforms->save(m_cachedir,  IT); 
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
       << std::endl << std::setw(20) << IT << " " << m_inverse_transforms->getShapeString() 
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


int GNodeLib::getFirstNodeIndexForGDMLAuxTargetLVName() const 
{
    const char* target_lvname = m_ok->getGDMLAuxTargetLVName() ; 

    std::vector<unsigned> nidxs ; 
    getNodeIndicesForLVName(nidxs, target_lvname); 

    int nidx = nidxs.size() > 0 ? nidxs[0] : -1 ; 

    LOG(info) 
        << " target_lvname " << target_lvname
        << " nidxs.size() " << nidxs.size()
        << " nidx " << nidx 
        ; 

    return nidx ; 
}

void GNodeLib::getNodeIndicesForLVName(std::vector<unsigned>& nidx, const char* lvname) const 
{
    if( lvname == NULL ) return ;  
    m_lvlist->getIndicesWithKey(nidx, lvname); 
}


void GNodeLib::dumpNodes(const std::vector<unsigned>& nidxs, const char* msg) const 
{
    LOG(info) << msg << " nidxs.size " << nidxs.size() ; 
    for(unsigned i=0 ; i < nidxs.size() ; i++)
    {
        unsigned nidx = nidxs[i]; 
        const char* pv = getPVName(nidx); 
        const char* lv = getLVName(nidx); 

        glm::vec4 ce = getCE(nidx);

        std::cout 
            << " i " << std::setw(5) << i 
            << " nidx " << std::setw(6) << nidx 
            << " pv " << std::setw(50) << pv 
            << " lv " << std::setw(50) << lv 
            << gpresent( "ce ", ce )
            ;
    }
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
GNodeLib::setRootVolume
------------------------

Require to defer GNodeLib::add of volumes until 
after GInstancer identity labelling, hence need
to hold on to root in the meantime.

Canonically invoked from X4PhysicalVolume::convertStructure/GGeo::setRootVolume

**/

void GNodeLib::setRootVolume(const GVolume* root)
{
    m_root = root ; 
}
const GVolume* GNodeLib::getRootVolume() const 
{
   return m_root ;  
}
 


/**
GNodeLib::addVolume (precache)
--------------------------------

Collects all volume information.

The triplet identity is only available on the volumes after 
GInstancer does the recursive labelling. So volume collection
is now done by GInstancer::collectNodes_r rather than the former 
X4PhysicalVolume::convertStructure.

**/

void GNodeLib::addVolume(const GVolume* volume)
{
    unsigned index = volume->getIndex(); 
    m_volumes.push_back(volume);
    assert( m_volumes.size() - 1 == index && "indices of the geometry volumes added to GNodeLib must follow the sequence : 0,1,2,... " ); // formerly only for m_test
    m_volumemap[index] = volume ; 

    glm::mat4 transform = volume->getTransformMat4();
    m_transforms->add(transform);  

    glm::mat4 inverse_transform = volume->getInverseTransformMat4();
    m_inverse_transforms->add(inverse_transform);  


    nbbox* bb = volume->getVerticesBBox(); 
    glm::vec4 min(bb->min, 1.f); 
    glm::vec4 max(bb->max, 1.f); 
    m_bounding_box->add( min, max); 

    glm::vec4 ce = bb->ce(); 
    m_center_extent->add(ce); 

    m_lvlist->add(volume->getLVName()); 
    m_pvlist->add(volume->getPVName()); 
    // NB added in tandem, so same counts and same index as the volumes  

    glm::uvec4 id = volume->getIdentity(); 
    m_identity->add(id);

    glm::uvec4 ni = volume->getNodeInfo(); 
    m_nodeinfo->add(ni);

    const GVolume* check = getVolume(index);
    assert(check == volume);

    m_num_volumes += 1 ; 

    bool is_sensor = volume->hasSensorIndex();
    if(is_sensor)
    {
        m_sensor_volumes.push_back(volume); 
        m_sensor_identity.push_back(id); 
        m_num_sensors += 1 ; 
    }
}

/**
GNodeLib::getNumSensorVolumes (precache and postcache)
---------------------------------------------------------
**/

unsigned GNodeLib::getNumSensorVolumes() const 
{
    return m_num_sensors ; 
}


/**
GNodeLib::getSensorVolume (precache only)
-------------------------------------------

**/

const GVolume* GNodeLib::getSensorVolume(unsigned sensorIndex) const 
{
    return m_loaded ? NULL : m_sensor_volumes[sensorIndex-1];  // 1-based sensorIndex
}

/**
GNodeLib::getSensorIdentity (precache and postcache)
------------------------------------------------------

**/
glm::uvec4 GNodeLib::getSensorIdentity(unsigned sensorIndex) const 
{
    glm::uvec4 sid = m_sensor_identity[sensorIndex-1];  // 1-based sensorIndex
    return sid ; 
}

unsigned GNodeLib::getSensorIdentityStandin(unsigned sensorIndex) const 
{
    glm::uvec4 sid = getSensorIdentity(sensorIndex); 
    return sid.y ;  // Opticks triplet Identifier
}



std::string GNodeLib::reportSensorVolumes(const char* msg) const 
{
    unsigned numSensorVolumes = getNumSensorVolumes(); 
    std::stringstream ss ; 
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

    if(m_loaded == false)
    {
        for(unsigned i=0 ; i < numSensorVolumes ; i++)
        {
            unsigned sensorIndex = 1 + i ; // 1-based 
            const GVolume* sensor = getSensorVolume(sensorIndex) ; 
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
                << " sensorIndex " << std::setw(6) << sensorIndex
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
    else
    {
        for(unsigned i=0 ; i < numSensorVolumes ; i++)
        {
            glm::uvec4 id = m_sensor_identity[i] ; 
            unsigned nidx = id.x ; 
            const char* pvname = getPVName(nidx);  
            std::cout 
                << std::setw(20) << glm::to_string(id) 
                << " " << pvname 
                << std::endl
                ; 
        }
 
    }
}


/**
GNodeLib::getSensorPlacements
------------------------------

TODO: eliminate the outer_volume kludge 

When outer_volume = true the placements returned are not 
those of the sensors themselves but rather those of the 
outer volumes of the instances that contain the sensors.

That is probably a kludge needed because it is the 
CopyNo of the  outer volume that carries the sensorId
for JUNO.  Need a way of getting that from the actual placed
sensor volume in detector specific code, not here.

**/

void GNodeLib::getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const 
{
    unsigned numSensorVolumes = getNumSensorVolumes(); 
    for(unsigned i=0 ; i < numSensorVolumes ; i++)
    {
        unsigned sensorIndex = 1 + i ; // 1-based
        const GVolume* sensor = getSensorVolume(sensorIndex) ; 
        assert(sensor); 

        void* origin = NULL ; 

        if(outer_volume) 
        {
            const GVolume* outer = sensor->getOuterVolume() ; 
            assert(outer); 
            origin = outer->getOriginNode() ;  
        } 
        else
        {
            origin = sensor->getOriginNode() ;  
        } 

        placements.push_back(origin); 
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



NPY<float>* GNodeLib::getTransforms() const 
{
    return m_transforms ; 
}
NPY<float>* GNodeLib::getInverseTransforms() const 
{
    return m_inverse_transforms ; 
}
NPY<float>* GNodeLib::getBoundingBox() const 
{
    return m_bounding_box ; 
}
NPY<float>* GNodeLib::getCenterExtent() const 
{
    return m_center_extent ; 
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
glm::mat4 GNodeLib::getInverseTransform(unsigned index) const 
{
    bool index_ok = index < m_num_volumes ; 
    if(!index_ok)
       LOG(fatal) 
           << " FATAL : index out of range "
           << " index " << index 
           << " m_num_volumes " << m_num_volumes
           ;
    assert( index_ok ); 
    glm::mat4 it = m_inverse_transforms->getMat4(index) ; 
    return it ;  
}





glm::vec4 GNodeLib::getCE(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::vec4 ce = m_center_extent->getQuad_(index) ; 
    return ce ;  
}
glm::uvec4 GNodeLib::getIdentity(unsigned index) const 
{
    bool expect = index < m_num_volumes ;
    if(!expect) LOG(error) << " index " << index  << " num_volumes " << m_num_volumes ; 
    assert( expect ); 
    //glm::uvec4 id = m_identity->getQuad(index) ; see notes/issues/triplet-id-loosing-offset-index-in-NPY.rst
    glm::uvec4 id = m_identity->getQuad_(index) ; 
    return id ;  
}
glm::uvec4 GNodeLib::getNRPO(unsigned nidx) const 
{
    glm::uvec4 id = getIdentity(nidx); 
    unsigned triplet = id.y ;
    unsigned ridx = OpticksIdentity::RepeatIndex(triplet) ; 
    unsigned pidx = OpticksIdentity::PlacementIndex(triplet) ; 
    unsigned oidx = OpticksIdentity::OffsetIndex(triplet) ; 
    glm::uvec4 nrpo(nidx,ridx,pidx,oidx);  
    return nrpo ; 
}



glm::uvec4 GNodeLib::getNodeInfo(unsigned index) const 
{
    assert( index < m_num_volumes ); 
    glm::uvec4 ni = m_nodeinfo->getQuad_(index) ; 
    return ni ;  
}



void GNodeLib::getBB(unsigned index, glm::vec4& mn, glm::vec4& mx ) const 
{
    assert( index < m_num_volumes ); 
    mn = m_bounding_box->getQuad_(index, 0); 
    mx = m_bounding_box->getQuad_(index, 1); 
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

    std::vector<int> targets ; 
    targets.push_back(m_ok->getTarget());         // --target
    targets.push_back(m_ok->getDomainTarget());   // --domaintarget
    targets.push_back(m_ok->getGenstepTarget());  // --gensteptarget
    targets.push_back(cursor); 

    LOG(info) 
        << msg  
        << " num_volumes " << num_volumes 
        << " --target " << m_ok->getTarget() 
        << " --domaintarget " << m_ok->getDomainTarget() 
        << " --gensteptarget " << m_ok->getGenstepTarget() 
        << " cursor " << cursor 
        ;

    LOG(info) << "first volumes "  ; 
    for(unsigned i=0 ; i < std::min(num_volumes, 20u) ; i++) 
    {    
        std::cout 
            << ( int(i) == cursor ? " **" : "   " ) 
            << descVolume(i)
            ;
    }    

    LOG(info) << "targetted volumes(**) OR volumes with extent greater than " << extent_cut_mm << " mm " ; 
    for(unsigned i=0 ; i < num_volumes ; i++) 
    {    
        glm::vec4 ce = getCE(i);

        bool is_target_volume = std::count(targets.begin(), targets.end(), int(i)) > 0 ; 

        if(ce.w > extent_cut_mm || is_target_volume )
        std::cout 
            << ( is_target_volume ? " **" : "   " ) 
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





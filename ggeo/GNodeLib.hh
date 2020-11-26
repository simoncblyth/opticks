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

#pragma once

#include <map>
#include <string>
#include <vector>
#include "plog/Severity.h"

template <typename T> class NPY ; 
struct nbbox ; 

class Opticks ; 

class GVolume ; 
class GNode ; 
class GNodeLib ; 
class GItemList ; 

class GTreePresent ; 


#include "GGEO_API_EXPORT.hh"

/*

GNodeLib
===========

* collection of GVolume/GNode instances with access by index.
* GGeo resident m_nodelib of instanciated in GGeo::initLibs

* **NB the full tree is not persisted currently, only arrays**

Initially was primarily a pre-cache operator, but access to pv/lv names also 
relevant post-cache.

There is now one canonical m_nodelib instance that is 
populated from GGeo::add(GVolume*) (Formerly GScene was another).

See also:

ggeo/tests/GNodeLibTest.cc
    loads the persisted GNodeLib and dumps node info 

ana/GNodeLib.py 
    loads the persisted GNodeLib and dumps node info

*/

class GGEO_API GNodeLib 
{
       friend class GGeo     ;  // for save 
       friend class GScene   ;  // for save 
    private:
       static const char* PV ; 
       static const char* LV ; 
       static const char* TR ; 
       static const char* IT ; 
       static const char* CE ; 
       static const char* BB ; 
       static const char* ID ; 
       static const char* NI ; 
    public:
        static const plog::Severity LEVEL ; 
        static const char* RELDIR ; 
        static const char* CacheDir(const Opticks* ok);
        static GNodeLib* Load(Opticks* ok);
        GNodeLib(Opticks* ok); 
    private:
        GNodeLib(Opticks* ok, bool loading); 
    public:
        std::string getShapeString() const ; 
        std::string desc() const ; 
        void dump(const char* msg="GNodeLib::dump") const ;
        void Dump(const char* msg="GNodeLib::Dump") const ;
    private:
        void save() const ;
        void init();
        unsigned initNumVolumes() const ;
        unsigned initSensorIdentity() ;

        GItemList*   getPVList(); 
        GItemList*   getLVList(); 
    public:
        // need to defer adding volumes after identity labelling, so need to hold onto root
        void           setRootVolume(const GVolume* root); 
        const GVolume* getRootVolume() const ; 
        Opticks*       getOpticks() const ; 
    public:
        unsigned     getNumPV() const ;
        unsigned     getNumLV() const ;
        void         addVolume(const GVolume*    volume);
        const GNode* getNode(unsigned index) const ; 
        const GVolume* getVolume(unsigned index) const ;  
        GVolume* getVolumeNonConst(unsigned index);

        const GVolume* getVolumeSimple(unsigned int index);  
        unsigned getNumVolumes() const ;
        std::vector<const GVolume*>& getVolumes() ; 
    public:
        const char* getPVName(unsigned int index) const ;
        const char* getLVName(unsigned int index) const ;
    public:
        int getFirstNodeIndexForGDMLAuxTargetLVName() const ;  // returns -1 when None configured
        void getNodeIndicesForLVName(std::vector<unsigned>& nidx, const char* lvname) const ;
        void dumpNodes(const std::vector<unsigned>& nidxs, const char* msg="GNodeLib::dumpNodes") const ;
    public:
        NPY<float>* getTransforms() const ; 
        NPY<float>* getInverseTransforms() const ; 
        NPY<float>* getBoundingBox() const ; 
        NPY<float>* getCenterExtent() const ; 
    public:
        unsigned getNumTransforms() const ; 
        glm::mat4 getTransform(unsigned index) const ;
        glm::mat4 getInverseTransform(unsigned index) const ;
        glm::uvec4 getIdentity(unsigned index) const ;
        glm::uvec4 getNRPO(unsigned index) const ;

        glm::uvec4 getNodeInfo(unsigned index) const ;
        glm::vec4 getCE(unsigned index) const ;
        void      getBB(unsigned index, glm::vec4& mn, glm::vec4& mx ) const ; 
        nbbox     getBBox(unsigned index) const ;


        std::string descVolume(unsigned index) const;
        void dumpVolumes(const std::map<std::string, int>& targets, const char* msg="GNodeLib::dumpVolumes", float extent_cut_mm=5000.f, int cursor=-1 ) const ; 

    public:
        unsigned         getNumSensorVolumes() const ;
        const GVolume*   getSensorVolume(unsigned sensorIndex) const ;
        glm::uvec4       getSensorIdentity(unsigned sensorIndex) const ;
        unsigned         getSensorIdentityStandin(unsigned sensorIndex) const ;
        std::string      reportSensorVolumes(const char* msg) const ; 
        void             dumpSensorVolumes(const char* msg="GNodeLib::dumpSensorVolumes") const ; 
        void             getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const ; 
    public:
        unsigned         findContainerVolumeIndex(float x, float y, float z) const ;
    private:
        Opticks*                           m_ok ;  
        const char*                        m_keydir ; 
        bool                               m_loaded ; 
        const char*                        m_cachedir ; 
        const char*                        m_reldir ; 
    private:
        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
        NPY<float>*                        m_transforms ; 
        NPY<float>*                        m_inverse_transforms ; 
        NPY<float>*                        m_bounding_box ; 
        NPY<float>*                        m_center_extent ; 
        NPY<unsigned>*                     m_identity ; 
        NPY<unsigned>*                     m_nodeinfo ; 
    private:
        GTreePresent*                      m_treepresent ; 
        unsigned                           m_num_volumes ; 
        std::vector<glm::uvec4>            m_sensor_identity ; 
        unsigned                           m_num_sensors ; 
    private:
        std::map<unsigned int, const GVolume*>    m_volumemap ; 
        std::vector<const GVolume*>               m_volumes ; 
        std::vector<const GVolume*>               m_sensor_volumes ; 
    private:
        const GVolume*                            m_root ;  



};
 


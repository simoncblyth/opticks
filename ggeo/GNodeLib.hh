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
* **NB currently only pv/lv names are persisted, not the volumes/nodes**
* the merged meshes and analytic information is of course persisted

The analytic and test ctor arguments determine the name of the 
persisting directory.

Initially was primarily a pre-cache operator, but access to pv/lv names also 
relevant post-cache.

There is now one canonical m_nodelib instance that is 
populated from GGeo::add(GVolume*) 
(Formerly GScene was another).

*/



class GGEO_API GNodeLib 
{
        friend class GGeo     ;  // for save 
        friend class GScene   ;  // for save 
    public:
        static const plog::Severity LEVEL ; 
        static const char* RELDIR ; 
        static const char* CacheDir(const Opticks* ok);
        static GNodeLib* Load(Opticks* ok);
        GNodeLib(Opticks* ok); 
    private:
        GNodeLib(Opticks* ok, bool loading); 
    public:
        std::string desc() const ; 
        void dump(const char* msg="GNodeLib::dump") const ;
    private:
        void save() const ;
        void init();
        GItemList*   getPVList(); 
        GItemList*   getLVList(); 

    public:
        unsigned getNumPV() const ;
        unsigned getNumLV() const ;
        void add(const GVolume*    volume);
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
        unsigned        addSensorVolume(const GVolume* volume) ;
        unsigned        getNumSensorVolumes() const ;
        const GVolume*  getSensorVolume(unsigned sensorIndex) const ;
        std::string     reportSensorVolumes(const char* msg) const ; 
        void            dumpSensorVolumes(const char* msg) const ; 
        void            getSensorPlacements(std::vector<void*>& placements) const ; 
    private:
        Opticks*                           m_ok ;  
        bool                               m_loading ; 
        const char*                        m_cachedir ; 
        const char*                        m_reldir ; 
    private:
        GItemList*                         m_pvlist ; 
        GItemList*                         m_lvlist ; 
        NPY<float>*                        m_transforms ; 
        NPY<float>*                        m_bounding_box ; 
        NPY<float>*                        m_center_extent ; 
    private:
        GTreePresent*                       m_treepresent ; 
    private:
        std::map<unsigned int, const GVolume*>    m_volumemap ; 
        std::vector<const GVolume*>               m_volumes ; 
        std::vector<const GVolume*>               m_sensor_volumes ; 



};
 


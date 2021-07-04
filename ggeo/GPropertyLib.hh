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

// for all (non-CUDA and CUDA) compilation
#define BOUNDARY_NUM_PROP 8
#define BOUNDARY_NUM_FLOAT4 2
#define BOUNDARY_NUM_MATSUR 4

#ifndef __CUDACC__
// only non-CUDA compilation

#include <map>
#include <string>
#include <vector>

#include <glm/fwd.hpp>
#include "plog/Severity.h"

class SLog ; 
template <typename T> class NPY ;
class NPYBase ; 

class BMeta ; 

class Opticks ; 
class OpticksResource ; 
class OpticksAttrSeq ; 

class GItemList ; 
template <typename T> class GDomain ;
template <typename T> class GProperty ;
template <typename T> class GPropertyMap ;

struct guint4 ; 


/**
GPropertyLib
==============


Subclasses
-----------

delta:ggeo blyth$ grep public\ GPropertyLib \*.hh

GBndLib.hh         :class GBndLib : public GPropertyLib {
GMaterialLib.hh    :class GMaterialLib : public GPropertyLib {
GScintillatorLib.hh:class GScintillatorLib : public GPropertyLib {
GSourceLib.hh      :class GSourceLib : public GPropertyLib {
GSurfaceLib.hh     :class GSurfaceLib : public GPropertyLib {


Lifecycle of GPropertyLib subclasses
------------------------------------

*ctor*
     constituent of GGeo instanciated in GGeo::init when running precache 
     or via GGeo::loadFromCache when running from cache

*init*
     invoked by *ctor*, sets up the keymapping and default properties 
     that are housed in GPropertyLib base

*add*
     from GGeo::loadFromG4DAE (ie in precache running only) 
     GMaterial instances are collected via AssimpGGeo::convertMaterials and GGeo::add

*close*
     GPropertyLib::close first invokes *sort* and then 
     serializes collected and potentially reordered objects via *createBuffer* 
     and *createNames* 

     * *close* is triggered by the first call to getIndex
     * after *close* no new materials can be added
     * *close* is canonically invoked by GBndLib::getOrCreate during AssimpGGeo::convertStructureVisit 
 
*save*
     buffer and names are written to cache by GPropertyLib::saveToCache

*load*
     static method that instanciates and populates via GPropertyLib::loadFromCache which
     reads in the buffer and names and then invokes *import*
     This allows operation from the cache without having to GGeo::loadFromG4DAE.

*import*
     reconstitutes the serialized objects and populates the collection of them
     TODO: digest checking the reconstitution

**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"
class GGEO_API GPropertyLib {
    public:
        static const plog::Severity LEVEL ;   
        static unsigned int UNSET ; 
        static unsigned int NUM_MATSUR ;    // number of material/surfaces in the boundary 
        static unsigned int NUM_PROP ; 
        static unsigned int NUM_FLOAT4 ; 
        static const char*  METANAME ; 
    public:
        static const char* material ; 
        static const char* surface ;
        static const char* source ; 
        static const char* bnd_ ;
    public:
        static BMeta* CreateAbbrevMeta(const std::vector<std::string>& names ); 
    public:
        const char*  getName(unsigned index) const ;
        unsigned getIndex(const char* shortname) const ;  // 0-based index of first matching name, UINT_MAX when no match
    public:
        // m_sensor_indices is a transient (non-persisted) vector of material/surface indices 
        bool isSensorIndex(unsigned index) const ; 
        void addSensorIndex(unsigned index); 
        unsigned getNumSensorIndices() const ;
        unsigned getSensorIndex(unsigned i) const ;
        void dumpSensorIndices(const char* msg) const ;
    public:
        void getIndicesWithNameEnding( std::vector<unsigned>& indices, const char* ending ) const ; 
    public:
        GPropertyLib(GPropertyLib* other, GDomain<double>* domain=NULL, bool optional=false);
        GPropertyLib(Opticks* ok, const char* type, bool optional=false);
        virtual ~GPropertyLib();
    public:
        unsigned    getUNSET();
        const char* getType() const ;
        const char* getComponentType() const ;
        Opticks*    getOpticks() const ; 
        std::string getCacheDir();
        std::string getPreferenceDir();
    public:
        void     dumpRaw(const char* msg="GPropertyLib::dumpRaw") const ;
        void                  addRaw( GPropertyMap<double>* pmap);
        unsigned              getNumRaw() const ;
        GPropertyMap<double>* getRaw(unsigned index) const ;
        GPropertyMap<double>* getRaw(const char* shortname) const ;
        void                  saveRaw();
        void                  loadRaw();
    public:
        void                  addRawOriginal(GPropertyMap<double>* pmap);
        unsigned              getNumRawOriginal() const ;
        GPropertyMap<double>* getRawOriginal(unsigned index) const ;
        GPropertyMap<double>* getRawOriginal(const char* shortname) const ;
        void                  saveRawOriginal();
        void                  loadRawOriginal();
    private:
        void     loadRaw( std::vector<GPropertyMap<double>*>& dst, const char* dirname_suffix, bool endswith ) ; 
    public:
        //void setOrder(std::map<std::string, unsigned int>& order);
        std::map<std::string, unsigned int>& getOrder(); 
    public:
        void getCurrentOrder(std::map<std::string, unsigned>& order ) ; 
    private:
        void init();
        void initOrder();
    public:
        // other classes need access to "shape" of the standardization
        static GDomain<double>* getDefaultDomain();
        static glm::vec4      getDefaultDomainSpec();
    public:
        void                 setStandardDomain(GDomain<double>* domain );
        GDomain<double>*      getStandardDomain();
        unsigned int         getStandardDomainLength();
        void                 dumpDomain(const char* msg="GPropertyLib::dumpDomain");
        void                 dumpNames(const char* msg="GPropertyLib::dumpNames") const  ; 
    public:
        GPropertyMap<double>* getDefaults();
        GProperty<double>*    getDefaultProperty(const char* name);
    public:
        // pure virtuals that need to be implemented in concrete subclasses
        virtual void defineDefaults(GPropertyMap<double>* defaults) = 0 ; 
        virtual void import() = 0 ; 
        virtual void sort() = 0 ; 
        virtual NPY<double>* createBuffer() = 0;
        virtual BMeta*      createMeta() = 0;
        virtual GItemList*  createNames() = 0;
    public:
        virtual void beforeClose() ;   // dont force an implemnetation, using empty dummy, but allow override 
    public:
        //GProperty<double>*    getItemProperty(const char* item, const char* pname) const ;
    public:
        GProperty<double>*    getPropertyOrDefault(GPropertyMap<double>* pmap, const char* pname);
        GProperty<double>*    getProperty(GPropertyMap<double>* pmap, const char* dkey) const ;
        GProperty<double>*    makeConstantProperty(double value);
        GProperty<double>*    makeRampProperty();
    public:
        void setKeyMap(const char* spec);
        const char* getLocalKey(const char* dkey) const ; // map standard -> local keys 
    public:
        void checkBufferCompatibility(unsigned int nk, const char* msg="GPropertyLib::checkBufferCompatibility");
    public:
        std::map<unsigned int, std::string> getNamesMap(); 
    public:
        NPY<unsigned int>* createUint4Buffer(std::vector<guint4>& vec);
        void importUint4Buffer(std::vector<guint4>& vec, NPY<unsigned int>* ibuf );
    public:
        //
        // *close* serializes the objects into Buffer and Names, 
        // this is triggered by the first call to getIndex, 
        // which is canonically invoked by GBndLib::getOrCreate during AssimpGGeo::convertStructureVisit 
        //
        void close();
        void setClosed(bool closed=true);
        bool isClosed() const ;
        bool hasDomain() const ;

        void setValid(bool valid=true);
        bool isValid() const ;

        void setNoLoad(bool noload=true);
        bool isNoLoad() const ;

        bool isOptional() const ;

        std::string  getBufferName(const char* suffix=NULL);
        NPY<double>*  getBuffer() const ;

        template <typename T> T getBufferMeta(const char* key, const char* fallback) const ;


        BMeta*        getMeta() const  ;
        GItemList*    getNames() const ;
        void saveNames(const char* idpath, const char* reldir, const char* txtname) const ; 

        OpticksAttrSeq*    getAttrNames();
        std::string getAbbr(const char* key) const ;
    public:
       void saveToCache(NPYBase* buffer, const char* suffix); // for extra buffers
       void saveNames(const char* dir=NULL) const ; // defaults to IDPATH
       void saveToCache();
       void loadFromCache();
    public:
        void setBuffer(NPY<double>* buf);
        void setMeta(BMeta* meta);
        void setNames(GItemList* names);
    public:
        static void SelectPropertyMapsWithProperties(std::vector<GPropertyMap<double>*>& dst, const char* props, char delim, const std::vector<GPropertyMap<double>*>& src) ;
        void findRawMapsWithProperties(       std::vector<GPropertyMap<double>*>& dst, const char* props, char delim );
        void findRawOriginalMapsWithProperties( std::vector<GPropertyMap<double>*>& dst, const char* props, char delim );
    protected:
        SLog*                                m_log ; 
        Opticks*                             m_ok ; 
        OpticksResource*                     m_resource ; 
    protected:
        NPY<double>*                         m_buffer ; 
        BMeta*                               m_meta ; 
        OpticksAttrSeq*                      m_attrnames ; // attributed name list 
        GItemList*                           m_names ;     // simple name list 
    protected:
        const char*                          m_type ; 
        const char*                          m_comptype ; 
        GDomain<double>*                      m_standard_domain ;  
        bool                                 m_optional ; 
    private:
        GPropertyMap<double>*                 m_defaults ;  
        std::map<std::string, std::string>   m_keymap ;   
        bool                                 m_closed ;  
        bool                                 m_valid ;  
        bool                                 m_noload ;  
    private:
        std::vector<GPropertyMap<double>*>    m_raw ; 
        std::vector<GPropertyMap<double>*>    m_raw_original ; 

        std::vector<unsigned>                 m_sensor_indices ; 


};

#include "GGEO_TAIL.hh"


#endif



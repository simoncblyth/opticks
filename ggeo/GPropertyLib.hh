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


class SLog ; 
template <typename T> class NPY ;
class NPYBase ; 

class NMeta ; 


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

delta:ggeo blyth$ grep public\ GPropertyLib *.hh

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
        static unsigned int UNSET ; 
        static unsigned int NUM_MATSUR ;    // number of material/surfaces in the boundary 
        static unsigned int NUM_PROP ; 
        static unsigned int NUM_FLOAT4 ; 
        static const char* METANAME ; 
    public:
        static const char* material ; 
        static const char* surface ;
        static const char* source ; 
        static const char* bnd_ ;
    public:
        const char*  getName(unsigned index) const ;
        unsigned getIndex(const char* shortname);
    public:
        void getIndicesWithNameEnding( std::vector<unsigned>& indices, const char* ending ) const ; 
    public:
        GPropertyLib(GPropertyLib* other, GDomain<float>* domain=NULL);
        GPropertyLib(Opticks* ok, const char* type);
        virtual ~GPropertyLib();
    public:
        unsigned    getUNSET();
        const char* getType();
        const char* getComponentType();
        Opticks*    getOpticks(); 
        std::string getCacheDir();
        std::string getPreferenceDir();
    public:
        void dumpRaw(const char* msg="GPropertyLib::dumpRaw");
        void addRaw(GPropertyMap<float>* pmap);
        unsigned int getNumRaw();
        GPropertyMap<float>* getRaw(unsigned int index);
        GPropertyMap<float>* getRaw(const char* shortname);
        void saveRaw();
        void loadRaw();
    public:
        //void setOrder(std::map<std::string, unsigned int>& order);
        std::map<std::string, unsigned int>& getOrder(); 
    private:
        void init();
        void initOrder();
    public:
        // other classes need access to "shape" of the standardization
        static GDomain<float>* getDefaultDomain();
        static glm::vec4      getDefaultDomainSpec();
    public:
        void                 setStandardDomain(GDomain<float>* domain );
        GDomain<float>*      getStandardDomain();
        unsigned int         getStandardDomainLength();
        void                 dumpDomain(const char* msg="GPropertyLib::dumpDomain");
        void                 dumpNames(const char* msg="GPropertyLib::dumpNames") const  ; 
    public:
        GPropertyMap<float>* getDefaults();
        GProperty<float>*    getDefaultProperty(const char* name);
    public:
        // pure virtuals that need to be implemented in concrete subclasses
        virtual void defineDefaults(GPropertyMap<float>* defaults) = 0 ; 
        virtual void import() = 0 ; 
        virtual void sort() = 0 ; 
        virtual NPY<float>* createBuffer() = 0;
        virtual NMeta*      createMeta() = 0;
        virtual GItemList*  createNames() = 0;
    public:
        virtual void beforeClose() ;   // dont force an implemnetation, using empty dummy, but allow override 
    public:
        //GProperty<float>*    getItemProperty(const char* item, const char* pname) const ;
    public:
        GProperty<float>*    getPropertyOrDefault(GPropertyMap<float>* pmap, const char* pname);
        GProperty<float>*    getProperty(GPropertyMap<float>* pmap, const char* dkey);
        GProperty<float>*    makeConstantProperty(float value);
        GProperty<float>*    makeRampProperty();
    public:
        void setKeyMap(const char* spec);
        const char* getLocalKey(const char* dkey); // map standard -> local keys 
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


        std::string  getBufferName(const char* suffix=NULL);
        NPY<float>*  getBuffer();
        NMeta*       getMeta() const  ;
        GItemList*   getNames() const ;
        OpticksAttrSeq*    getAttrNames();
        std::string getAbbr(const char* key);
    public:
       void saveToCache(NPYBase* buffer, const char* suffix); // for extra buffers
       void saveNames(const char* dir=NULL) const ; // defaults to IDPATH
       void saveToCache();
       void loadFromCache();
    public:
        void setBuffer(NPY<float>* buf);
        void setMeta(NMeta* meta);
        void setNames(GItemList* names);
    protected:
        SLog*                                m_log ; 
        Opticks*                             m_ok ; 
        OpticksResource*                     m_resource ; 
    protected:
        NPY<float>*                          m_buffer ; 
        NMeta*                               m_meta ; 
        OpticksAttrSeq*                      m_attrnames ; // attributed name list 
        GItemList*                           m_names ;     // simple name list 
    protected:
        const char*                          m_type ; 
        const char*                          m_comptype ; 
        GDomain<float>*                      m_standard_domain ;  
    private:
        GPropertyMap<float>*                 m_defaults ;  
        std::map<std::string, std::string>   m_keymap ;   
        bool                                 m_closed ;  
        bool                                 m_valid ;  
        bool                                 m_noload ;  
    private:
        std::vector<GPropertyMap<float>*>    m_raw ; 
};

#include "GGEO_TAIL.hh"


#endif



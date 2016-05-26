#pragma once

// for all (non-CUDA and CUDA) compilation
#define BOUNDARY_NUM_PROP 8
#define BOUNDARY_NUM_FLOAT4 2

#define BOUNDARY_NUM_MATSUR 4


#ifndef __CUDACC__
// only non-CUDA compilation

#include <map>
#include <string>
#include <cstring>

#include <vector>
#include <glm/glm.hpp>

#include "GDomain.hh"
#include "GPropertyMap.hh"
#include "GVector.hh"

template <typename T> class NPY ;
class NPYBase ; 

class OpticksResource ; 
class GCache ; 
class GItemList ; 
class GAttrSeq ; 


/*
See GMaterialLib.hh for description of lifecycle of all GPropertyLib subclasses

delta:ggeo blyth$ grep public\ GPropertyLib *.hh

GBndLib.hh         :class GBndLib : public GPropertyLib {
GMaterialLib.hh    :class GMaterialLib : public GPropertyLib {
GScintillatorLib.hh:class GScintillatorLib : public GPropertyLib {
GSourceLib.hh      :class GSourceLib : public GPropertyLib {
GSurfaceLib.hh     :class GSurfaceLib : public GPropertyLib {

*/

class GPropertyLib {
    public:
        static unsigned int UNSET ; 
        static unsigned int NUM_MATSUR ;    // number of material/surfaces in the boundary 
        static unsigned int NUM_PROP ; 
        static unsigned int NUM_FLOAT4 ; 
    public:
        static const char* material ; 
        static const char* surface ;
        static const char* source ; 
        static const char* bnd ;
    public:
        const char*  getName(unsigned int index);
        unsigned int getIndex(const char* shortname);
    public:
        GPropertyLib(GCache* cache, const char* type);
        virtual ~GPropertyLib();
    public:
        const char* getType();
        const char* getComponentType();
        std::string getCacheDir();
        std::string getPreferenceDir();
    public:
        void dumpRaw(const char* msg="GPropertyLib::dumpRaw");
        void addRaw(GPropertyMap<float>* pmap);
        unsigned int getNumRaw();
        GPropertyMap<float>* getRaw(unsigned int index);
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
        GDomain<float>*      getStandardDomain();
        unsigned int         getStandardDomainLength();
    public:
        GPropertyMap<float>* getDefaults();
        GProperty<float>*    getDefaultProperty(const char* name);
    public:
        // pure virtuals that need to be implemented in concrete subclasses
        virtual void defineDefaults(GPropertyMap<float>* defaults) = 0 ; 
        virtual void import() = 0 ; 
        virtual void sort() = 0 ; 
        virtual NPY<float>* createBuffer() = 0;
        virtual GItemList*  createNames() = 0;
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
        bool isClosed();
        void setValid(bool valid=true);
        bool isValid();

        std::string  getBufferName(const char* suffix=NULL);
        NPY<float>*  getBuffer();
        GItemList*   getNames();
        GAttrSeq*    getAttrNames();
    public:
       void saveToCache(NPYBase* buffer, const char* suffix); // for extra buffers
       void saveToCache();
       void loadFromCache();
    public:
        void         setBuffer(NPY<float>* buf);
        void         setNames(GItemList* names);
    protected:
        GCache*                              m_cache ; 
        OpticksResource*                     m_resource ; 
        NPY<float>*                          m_buffer ; 
        GAttrSeq*                            m_attrnames ; // attributed name list 
        GItemList*                           m_names ;     // simple name list 
        const char*                          m_type ; 
        const char*                          m_comptype ; 
        GDomain<float>*                      m_standard_domain ;  
    private:
        GPropertyMap<float>*                 m_defaults ;  
        std::map<std::string, std::string>   m_keymap ;   
        bool                                 m_closed ;  
        bool                                 m_valid ;  
    private:
        std::vector<GPropertyMap<float>*>    m_raw ; 
};

inline GPropertyLib::GPropertyLib(GCache* cache, const char* type) 
     :
     m_cache(cache),
     m_resource(NULL),
     m_buffer(NULL),
     m_attrnames(NULL),
     m_names(NULL),
     m_type(strdup(type)),
     m_comptype(NULL),
     m_standard_domain(NULL),
     m_defaults(NULL),
     m_closed(false),
     m_valid(true)
{
     init();
}




inline const char* GPropertyLib::getType()
{
    return m_type ; 
}

inline const char* GPropertyLib::getComponentType()
{
    return m_comptype ; 
}




inline GPropertyLib::~GPropertyLib()
{
}

inline GDomain<float>* GPropertyLib::getStandardDomain()
{
    return m_standard_domain ;
}

/*
inline void GPropertyLib::setOrder(std::map<std::string, unsigned int>& order)
{
    m_order = order ; 
}
*/

inline GPropertyMap<float>* GPropertyLib::getDefaults()
{
    return m_defaults ;
}

inline void GPropertyLib::setBuffer(NPY<float>* buf)
{
    m_buffer = buf ;
}
inline NPY<float>* GPropertyLib::getBuffer()
{
    return m_buffer ;
}

inline GItemList* GPropertyLib::getNames()
{
    return m_names ;
}
inline GAttrSeq* GPropertyLib::getAttrNames()
{
    return m_attrnames ;
}


inline void GPropertyLib::setClosed(bool closed)
{
    m_closed = closed ; 
}
inline bool GPropertyLib::isClosed()
{
    return m_closed ; 
}

inline void GPropertyLib::setValid(bool valid)
{
    m_valid = valid ; 
}
inline bool GPropertyLib::isValid()
{
    return m_valid ; 
}

inline unsigned int GPropertyLib::getNumRaw()
{
    return m_raw.size();
}


#endif



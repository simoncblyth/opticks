#pragma once

#include <map>
#include <string>

#include <vector>

#include "GDomain.hh"
#include "GPropertyMap.hh"
#include "GVector.hh"

template <typename T> class NPY ;
class NPYBase ; 

class GCache ; 
class GItemList ; 
class GAttrSeq ; 


/*
See GMaterialLib.hh for description of lifecycle of all GPropertyLib subclasses
*/

class GPropertyLib {
    public:
        static unsigned int UNSET ; 
        static unsigned int NUM_QUAD ; 
        static unsigned int NUM_PROP ; 
        static unsigned int DOMAIN_LENGTH ; 
        static float        DOMAIN_LOW ; 
        static float        DOMAIN_HIGH ; 
        static float        DOMAIN_STEP ; 
    public:
        const char*  getName(unsigned int index);
        unsigned int getIndex(const char* shortname);
    public:
        GPropertyLib(GCache* cache, const char* type);
        virtual ~GPropertyLib();
    public:
        const char* getType();
        std::string getCacheDir();
        std::string getPreferenceDir();
    public:
        //void setOrder(std::map<std::string, unsigned int>& order);
        std::map<std::string, unsigned int>& getOrder(); 
    private:
        void init();
        void initOrder();
    public:
        GDomain<float>*      getStandardDomain();
        unsigned int         getStandardDomainLength();
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
        // another classes need access to "shape" of the standardization
        static GDomain<float>* getDefaultDomain();
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
        NPY<float>*                          m_buffer ; 
        GAttrSeq*                            m_attrnames ; // attributed name list 
        GItemList*                           m_names ;     // simple name list 
        const char*                          m_type ; 
        GDomain<float>*                      m_standard_domain ;  

    private:
        GPropertyMap<float>*                 m_defaults ;  
        std::map<std::string, std::string>   m_keymap ;   
        bool                                 m_closed ;  
};

inline GPropertyLib::GPropertyLib(GCache* cache, const char* type) 
     :
     m_cache(cache),
     m_buffer(NULL),
     m_attrnames(NULL),
     m_names(NULL),
     m_type(strdup(type)),
     m_standard_domain(NULL),
     m_defaults(NULL),
     m_closed(false)
{
     init();
}


inline const char* GPropertyLib::getType()
{
    return m_type ; 
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



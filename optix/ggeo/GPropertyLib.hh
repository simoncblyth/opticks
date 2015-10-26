#pragma once

#include <map>
#include <string>

#include "GDomain.hh"
#include "GPropertyMap.hh"

class GCache ; 

class GPropertyLib {
    public:
        static unsigned int DOMAIN_LENGTH ; 
        static float        DOMAIN_LOW ; 
        static float        DOMAIN_HIGH ; 
        static float        DOMAIN_STEP ; 
    public:
        GPropertyLib(GCache* cache);
        virtual ~GPropertyLib();

        void setStandardDomain(GDomain<float>* standard_domain);
        void setDefaults(GPropertyMap<float>* defaults);
    private:
        void init();
    public:
        GDomain<float>*      getStandardDomain();
        unsigned int         getStandardDomainLength();
        GPropertyMap<float>* getDefaults();
        GProperty<float>*    getDefaultProperty(const char* name);
    public:
        // defaults need to be set in concrete subclass
        virtual void defineDefaults(GPropertyMap<float>* defaults) = 0 ; 
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

    private:
        GCache*              m_cache ; 
        GDomain<float>*      m_standard_domain ;  
        GPropertyMap<float>* m_defaults ;  
        std::map<std::string, std::string>   m_keymap ; //  
};

inline GPropertyLib::GPropertyLib(GCache* cache) 
     :
     m_cache(cache),
     m_standard_domain(NULL),
     m_defaults(NULL)
{
     init();
}


inline GPropertyLib::~GPropertyLib()
{
}


inline void GPropertyLib::setStandardDomain(GDomain<float>* standard_domain)
{
    m_standard_domain = standard_domain ; 
}

inline GDomain<float>* GPropertyLib::getStandardDomain()
{
    return m_standard_domain ;
}


inline void GPropertyLib::setDefaults(GPropertyMap<float>* defaults)
{
    m_defaults = defaults ;
}

inline GPropertyMap<float>* GPropertyLib::getDefaults()
{
    return m_defaults ;
}



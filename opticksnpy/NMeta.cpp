
#include <boost/lexical_cast.hpp>

#include "NMeta.hpp"
#include "NJS.hpp"

#include "PLOG.hh"


NMeta::NMeta()
    :
    m_js(new NJS)
{
}


nlohmann::json& NMeta::js()
{
    return m_js->js() ; 
}  
 

void NMeta::load(const char* path)
{
    m_js->read(path);
}
void NMeta::load(const char* dir, const char* name)
{
    m_js->read(dir, name);
}

NMeta* NMeta::Load(const char* path0)
{
    NMeta* m = new NMeta ;
    m->load(path0); 
    return m ; 
}
NMeta* NMeta::Load(const char* dir, const char* name)
{
    NMeta* m = new NMeta ;
    m->load(dir, name); 
    return m ; 
}

void NMeta::save(const char* path) const 
{
    m_js->write(path);
}
void NMeta::save(const char* dir, const char* name) const
{
    m_js->write(dir, name);
}

void NMeta::dump() const 
{
    nlohmann::json& js = m_js->js();
    LOG(info) << js.dump(4) ; 
}


void NMeta::set(const char* name, NMeta* obj)
{
    nlohmann::json& js = m_js->js();
    js[name] = obj->js(); 
}


template <typename T>
void NMeta::set(const char* name, T value)
{
    nlohmann::json& js = m_js->js();
    js[name] = value ; 
}

template <typename T>
T NMeta::get(const char* name, const char* fallback) const 
{
    nlohmann::json& js = m_js->js();
    return js.count(name) == 1 ? js[name].get<T>() : boost::lexical_cast<T>(fallback);
}

template <typename T>
T NMeta::get(const char* name) const 
{
    nlohmann::json& js = m_js->js();
    assert( js.count(name) == 1  );
    return js[name].get<T>() ;
}


template NPY_API void NMeta::set(const char* name, bool value);
template NPY_API void NMeta::set(const char* name, int value);
template NPY_API void NMeta::set(const char* name, unsigned int value);
template NPY_API void NMeta::set(const char* name, std::string value);
template NPY_API void NMeta::set(const char* name, float value);
template NPY_API void NMeta::set(const char* name, char value);


template NPY_API bool         NMeta::get(const char* name) const ;
template NPY_API int          NMeta::get(const char* name) const ;
template NPY_API unsigned int NMeta::get(const char* name) const ;
template NPY_API std::string  NMeta::get(const char* name) const ;
template NPY_API float        NMeta::get(const char* name) const ;
template NPY_API char         NMeta::get(const char* name) const ;


template NPY_API bool         NMeta::get(const char* name, const char* fallback) const ;
template NPY_API int          NMeta::get(const char* name, const char* fallback) const ;
template NPY_API unsigned int NMeta::get(const char* name, const char* fallback) const ;
template NPY_API std::string  NMeta::get(const char* name, const char* fallback) const ;
template NPY_API float        NMeta::get(const char* name, const char* fallback) const ;
template NPY_API char         NMeta::get(const char* name, const char* fallback) const ;


/*
template NPY_API void NMeta::set(const char* name, nlohmann::json value);
template NPY_API nlohmann::json  NMeta::get(const char* name) const ;
template NPY_API nlohmann::json  NMeta::get(const char* name, const char* fallback) const ;
*/



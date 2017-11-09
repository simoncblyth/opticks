
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>

#include <boost/lexical_cast.hpp>

#include "BFile.hh"
#include "NMeta.hpp"

#include "PLOG.hh"


NMeta::NMeta(const NMeta& other)
    :
    m_js(other.cjs())
{
}

NMeta::NMeta() 
    : 
    m_js()
{
}

nlohmann::json& NMeta::js()
{
    return m_js ; 
}  

const nlohmann::json& NMeta::cjs() const 
{
    return m_js ; 
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


void NMeta::load(const char* path)
{
    read(path);
}
void NMeta::load(const char* dir, const char* name)
{
    read(dir, name);
}

void NMeta::save(const char* path) const 
{
    write(path);
}
void NMeta::save(const char* dir, const char* name) const
{
    write(dir, name);
}



void NMeta::dump() const 
{
    LOG(info) << m_js.dump(4) ; 
}

void NMeta::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    std::cout << m_js.dump(4) << std::endl ; 
}




void NMeta::setObj(const char* name, NMeta* obj)
{
    m_js[name] = obj->js(); 
}

NMeta* NMeta::getObj(const char* name)
{
    nlohmann::json& this_js = m_js ;

    NMeta* obj = new NMeta ; 
    nlohmann::json& obj_js = obj->js();
    obj_js = this_js[name] ; 

    return obj ; 
}

void NMeta::updateKeys() 
{
    m_keys.clear();
    for (nlohmann::json::const_iterator it = m_js.begin(); it != m_js.end(); ++it) 
    {
        m_keys.push_back( it.key() );
    }
}

std::string NMeta::desc(unsigned wid)
{
    std::stringstream ss ; 
    ss << std::setw(wid) << m_js ; 
    return ss.str();
}

unsigned NMeta::getNumKeys() 
{
    updateKeys() ;
    return m_keys.size();
}

const char* NMeta::getKey(unsigned idx) const 
{
    assert( idx < m_keys.size() );
    return m_keys[idx].c_str() ; 
}


template <typename T>
void NMeta::set(const char* name, T value)
{
    m_js[name] = value ; 
}

template <typename T>
T NMeta::get(const char* name, const char* fallback) const 
{
    return m_js.count(name) == 1 ? m_js[name].get<T>() : boost::lexical_cast<T>(fallback);
}

template <typename T>
T NMeta::get(const char* name) const 
{
    assert( m_js.count(name) == 1  );
    return m_js[name].get<T>() ;
}

bool NMeta::hasItem(const char* name) const 
{
    return m_js.count(name) == 1 ;
}







void NMeta::read(const char* path0, const char* path1)
{
    std::string path = BFile::FormPath(path0, path1);

    LOG(debug) << "read from " << path ; 

    std::ifstream in(path.c_str(), std::ios::in);

    if(!in.is_open()) 
    {   
        LOG(debug) << "NMeta::read failed to open " << path ; 
        return ;
    }   
    in >> m_js ; 
}

void NMeta::write(const char* path0, const char* path1) const 
{
    std::string path = BFile::FormPath(path0, path1);

    std::string pdir = BFile::ParentDir(path.c_str());

    BFile::CreateDir(pdir.c_str()); 

    LOG(info) << "write to " << path ; 

    std::ofstream out(path.c_str(), std::ios::out);

    if(!out.is_open()) 
    {   
        LOG(fatal) << "NMeta::write failed to open" << path ; 
        return ;
    }   

    out << m_js ; 

    out.close();
}




template NPY_API void NMeta::set(const char* name, bool value);
template NPY_API void NMeta::set(const char* name, int value);
template NPY_API void NMeta::set(const char* name, unsigned int value);
template NPY_API void NMeta::set(const char* name, std::string value);
template NPY_API void NMeta::set(const char* name, float value);
template NPY_API void NMeta::set(const char* name, char value);
//template NPY_API void NMeta::set(const char* name, const char* value);


template NPY_API bool         NMeta::get(const char* name) const ;
template NPY_API int          NMeta::get(const char* name) const ;
template NPY_API unsigned int NMeta::get(const char* name) const ;
template NPY_API std::string  NMeta::get(const char* name) const ;
template NPY_API float        NMeta::get(const char* name) const ;
template NPY_API char         NMeta::get(const char* name) const ;
//template NPY_API const char*  NMeta::get(const char* name) const ;


template NPY_API bool         NMeta::get(const char* name, const char* fallback) const ;
template NPY_API int          NMeta::get(const char* name, const char* fallback) const ;
template NPY_API unsigned int NMeta::get(const char* name, const char* fallback) const ;
template NPY_API std::string  NMeta::get(const char* name, const char* fallback) const ;
template NPY_API float        NMeta::get(const char* name, const char* fallback) const ;
template NPY_API char         NMeta::get(const char* name, const char* fallback) const ;
//template NPY_API const char*  NMeta::get(const char* name, const char* fallback) const ;




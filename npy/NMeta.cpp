
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>


#ifdef _MSC_VER
#else
#include <unistd.h>
extern char **environ;
#endif

#include <boost/lexical_cast.hpp>

#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"
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


unsigned NMeta::size() const 
{
    return m_js.size() ; 
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


std::vector<std::string>& NMeta::getLines()
{
    if(m_lines.size() == 0 ) prepLines();
    return m_lines ;
}

void NMeta::dumpLines(const char* msg) 
{
    LOG(info) << msg ; 
    const std::vector<std::string>& lines = getLines(); 
    for(unsigned i=0 ; i < lines.size(); i++)
    {
        std::cout << lines[i] << std::endl ;   
    }
}

void NMeta::prepLines()
{
    m_lines.clear();
    for (nlohmann::json::const_iterator it = m_js.begin(); it != m_js.end(); ++it) 
    {
        std::string name = it.key(); 
        std::stringstream ss ;  
        ss 
             << std::fixed
             << std::setw(15) << name
             << " : " 
             << std::setw(15) << m_js[name.c_str()] 
             ;
        
        m_lines.push_back(ss.str());
    }
}



void NMeta::dump() const 
{
    LOG(info) << std::endl << m_js.dump(4) ; 
}
void NMeta::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    std::cout << m_js.dump(4) << std::endl ; 
}



void NMeta::append(NMeta* other) 
{
    if(!other) return ; 
    for (const auto &j : nlohmann::json::iterator_wrapper(other->js())) m_js[j.key()] = j.value();
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
void NMeta::add(const char* name, T value)
{
    m_js[name] = value ; 
}






template <typename T>
T NMeta::get(const char* name, const char* fallback) const 
{
    return m_js.count(name) == 1 ? m_js[name].get<T>() : boost::lexical_cast<T>(fallback);
}

/**

NMeta::getIntFromString
=========================

Many python stored verbosity appear as strings in the json, eg 248/meta.json:: 

    {"lvname": "World0xc15cfc0", "soname": "WorldBox0xc15cf40", "lvIdx": 248, "verbosity": "0", "resolution": "20", "poly": "IM", "height": 0}
                                                                                           ^^^
                                                                                           ^^^

But metadata reading code expects an int. This causes no problem for the untyped BParameter which stored 
everything as strings, but it does cause a problem with NMeta. Causing test fails of   
NCSGLoadTest, NScanTest, NSceneTest.

This method is a workaround for this.

**/

int NMeta::getIntFromString(const char* name, const char* fallback) const 
{
    // workaround for many verbosity being encoded as string
    std::string s = get<std::string>(name, "0");
    int f = BStr::atoi(fallback, 0);    
    int v = BStr::atoi(s.c_str(), f);    
    return v ;
}





template <typename T>
T NMeta::Get(const NMeta* meta, const char* name, const char* fallback) // static
{
    return meta ? meta->get<T>(name, fallback) : boost::lexical_cast<T>(fallback) ;
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


void NMeta::addEnvvar( const char* key ) 
{
    const char* val = SSys::getenvvar(key) ; 
    if( val ) 
    {
        std::string s = val ; 
        set<std::string>(key, s) ; 

    } 
} 

void NMeta::addEnvvarsWithPrefix( const char* prefix, bool trim ) 
{
    int i=0 ; 
    while(*(environ+i))
    {
       char* kv_ = environ[i++] ;  
       if(strncmp(kv_, prefix, strlen(prefix))==0)
       { 
           std::string kv = kv_ ; 

           size_t p = kv.find('=');  
           assert( p != std::string::npos) ; 

           std::string k = kv.substr(0,p); 
           std::string v = kv.substr(p+1);   

           std::string t = k.substr(strlen(prefix)); 
  
           LOG(info) << k << " : " << t << " : " << v   ;   

           set<std::string>( trim ? t.c_str() : k.c_str(), v) ; 
       }
    }      
}


void NMeta::appendString(const char* name, const std::string& value, const char* delim)
{
    bool found = false ; 
    std::string avalue ; 

    for (nlohmann::json::const_iterator it = m_js.begin(); it != m_js.end(); ++it) 
    {
        std::string n = it.key() ;
        if(n.compare(name)==0) 
        {
            found = true ; 
            std::stringstream ss ; 
            ss << m_js[n.c_str()].get<std::string>()  << delim  << value ; 
            avalue = ss.str(); 
            break ; 
        }
    }

    std::string v = found ? avalue : value ;     

    set<std::string>(name, v) ;
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
    if(path0 == NULL  && path1 == NULL)
    {
        LOG(fatal) << " NULL paths " ; 
        return ; 
    }
 
    std::string path = BFile::FormPath(path0, path1);

    std::string pdir = BFile::ParentDir(path.c_str());

    BFile::CreateDir(pdir.c_str()); 

    LOG(debug) << "write to " << path ; 

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
template NPY_API void NMeta::set(const char* name, double  value);
template NPY_API void NMeta::set(const char* name, char value);
//template NPY_API void NMeta::set(const char* name, const char* value);


template NPY_API void NMeta::add(const char* name, bool value);
template NPY_API void NMeta::add(const char* name, int value);
template NPY_API void NMeta::add(const char* name, unsigned int value);
template NPY_API void NMeta::add(const char* name, std::string value);
template NPY_API void NMeta::add(const char* name, float value);
template NPY_API void NMeta::add(const char* name, double value);
template NPY_API void NMeta::add(const char* name, char value);
//template NPY_API void NMeta::add(const char* name, const char* value);


template NPY_API bool         NMeta::get(const char* name) const ;
template NPY_API int          NMeta::get(const char* name) const ;
template NPY_API unsigned int NMeta::get(const char* name) const ;
template NPY_API std::string  NMeta::get(const char* name) const ;
template NPY_API float        NMeta::get(const char* name) const ;
template NPY_API double       NMeta::get(const char* name) const ;
template NPY_API char         NMeta::get(const char* name) const ;
//template NPY_API const char*  NMeta::get(const char* name) const ;


template NPY_API bool         NMeta::get(const char* name, const char* fallback) const ;
template NPY_API int          NMeta::get(const char* name, const char* fallback) const ;
template NPY_API unsigned int NMeta::get(const char* name, const char* fallback) const ;
template NPY_API std::string  NMeta::get(const char* name, const char* fallback) const ;
template NPY_API float        NMeta::get(const char* name, const char* fallback) const ;
template NPY_API double       NMeta::get(const char* name, const char* fallback) const ;
template NPY_API char         NMeta::get(const char* name, const char* fallback) const ;
//template NPY_API const char*  NMeta::get(const char* name, const char* fallback) const ;


template NPY_API bool         NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API int          NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API unsigned int NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API std::string  NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API float        NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API double       NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
template NPY_API char         NMeta::Get(const NMeta*, const char* name, const char* fallback) ; 
//template NPY_API const char*  NMeta::Get(const NMeta*,const char* name, const char* fallback) ; 



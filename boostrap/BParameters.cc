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

#include "BParameters.hh"

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>

#ifdef _MSC_VER
#else
#include <unistd.h>
extern char **environ;
#endif




#include "SSys.hh"
#include "BFile.hh"
#include "BList.hh"
#include "PLOG.hh"



BParameters::BParameters()
{
}

std::vector<std::string>& BParameters::getLines()
{
    if(m_lines.size() == 0 ) prepLines();
    return m_lines ;
}

const std::vector<std::pair<std::string,std::string> >& BParameters::getVec()
{
    return m_parameters ; 
}


std::string BParameters::getStringValue(const char* name) const 
{
    std::string value ; 
    for(VSS::const_iterator it=m_parameters.begin() ; it != m_parameters.end() ; it++)
    {
        std::string npar  = it->first ; 
        if(npar.compare(name)==0) value = it->second ; 
    }
    return value ;  
}




bool BParameters::load_(const char* path)
{
    bool exists = BFile::ExistsFile(path) ; 
    if(!exists) return false ; 
    BList<std::string, std::string>::load(&m_parameters, path);
    return true ; 
}
bool BParameters::load_(const char* dir, const char* name)
{
    bool exists = BFile::ExistsFile(dir, name) ; 
    if(!exists) return false ; 

    BList<std::string, std::string>::load(&m_parameters, dir, name);
    return true ; 
}
BParameters* BParameters::Load(const char* path)
{
    bool exists = BFile::ExistsFile(path) ; 
    if(!exists) return NULL ;  

    BParameters* p = new BParameters ;
    p->load_(path); 
    return p ; 
}
BParameters* BParameters::Load(const char* dir, const char* name)
{
    bool exists = BFile::ExistsFile(dir, name) ; 
    if(!exists) return NULL ; 

    BParameters* p = new BParameters ;
    p->load_(dir, name); 
    return p ; 
}



void BParameters::save(const char* path)
{
    BList<std::string, std::string>::save(&m_parameters, path);
}
void BParameters::save(const char* dir, const char* name)
{
    BList<std::string, std::string>::save(&m_parameters, dir, name);
}


void BParameters::dump()
{
    dump("BParameters::dump");  // handy for debugging::   (lldb) expr m_parameters->dump()
}

std::string BParameters::desc()
{
    prepLines();
    std::stringstream ss ; 
    ss << "BParameters numItems " << getNumItems() ; 
    for(VS::const_iterator it=m_lines.begin() ; it != m_lines.end() ; it++) ss << *it << " : " ;  
    return ss.str();
}


void BParameters::dump(const char* msg)
{
   prepLines();
   std::cout << msg << std::endl ; 
   for(VS::const_iterator it=m_lines.begin() ; it != m_lines.end() ; it++) std::cout << *it << std::endl ;  
}

void BParameters::prepLines()
{
    m_lines.clear();
    for(VSS::const_iterator it=m_parameters.begin() ; it != m_parameters.end() ; it++)
    {
        std::string name  = it->first ; 
        std::string value = it->second ; 

        std::stringstream ss ;  
        ss 
             << std::fixed
             << std::setw(15) << name
             << " : " 
             << std::setw(15) << value 
             ;
        
        m_lines.push_back(ss.str());
    }
}



unsigned BParameters::getNumItems()
{
   return m_parameters.size(); 
}


template <typename T>
void BParameters::set(const char* name, T value)
{
    std::string svalue = boost::lexical_cast<std::string>(value) ;
    bool found(false);
    bool startswith(false);

    for(VSS::iterator it=m_parameters.begin() ; it != m_parameters.end() ; it++)
    {
        std::string npar  = it->first ; 

        bool match = startswith ?  
                                   strncmp(npar.c_str(), name, strlen(name))==0 
                                :
                                   npar.compare(name)==0
                                ;

        if(match) 
        {
            std::string prior = it->second ; 
            LOG(debug) << "BParameters::set changing "
                      << name 
                      << " from " << prior 
                      << " to " << svalue 
                      ;
                  
            it->second = svalue ; 
            found = true ; 
        }
    }

    if(!found) add<T>(name, value) ;
 
}


template <typename T>
void BParameters::add(const char* name, T value)
{
    m_parameters.push_back(SS(name, boost::lexical_cast<std::string>(value) ));
}


void BParameters::addEnvvar( const char* key ) 
{
    const char* val = SSys::getenvvar(key) ; 
    if( val ) 
    {
        std::string s = val ; 
        add<std::string>(key, s) ; 

    } 
} 

void BParameters::addEnvvarsWithPrefix( const char* prefix ) 
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
   
           //std::cout << k << " : " << v << std::endl ;   

           add<std::string>(k.c_str(), v );   
       }
    }      
}



template <typename T>
void BParameters::append(const char* name, T value, const char* delim)
{
    std::string svalue = boost::lexical_cast<std::string>(value) ;
    bool found = false ; 
    for(VSS::iterator it=m_parameters.begin() ; it != m_parameters.end() ; it++)
    {
        std::string n  = it->first ; 
        if(n.compare(name)==0) 
        {
            found = true ; 
            std::stringstream ss ; 
            ss << it->second  << delim  << svalue ; 
            it->second = ss.str();
            break ; 
        }
    } 
    if(!found) add<T>(name, value) ;
}


void BParameters::append(BParameters* other)
{
    if(!other) return ; 

    const VSS& other_v = other->getVec();
    for(VSS::const_iterator it=other_v.begin() ; it != other_v.end() ; it++)
    {
        std::string name  = it->first ; 
        std::string value = it->second ; 
        m_parameters.push_back(SS(name, value));
    }
}

 


template <typename T>
T BParameters::get(const char* name) const 
{
    std::string value = getStringValue(name);
    if(value.empty())
    {
        LOG(fatal) << "BParameters::get " << name << " EMPTY VALUE "  ;
    }
    return boost::lexical_cast<T>(value);
}
 
 
template <typename T>
T BParameters::get(const char* name, const char* fallback) const 
{
    std::string value = getStringValue(name);
    if(value.empty())
    {
        value = fallback ;  
        LOG(debug) << "BParameters::get " << name << " value empty, using fallback value: " << fallback  ;
    }
    return boost::lexical_cast<T>(value);
}





template BRAP_API void BParameters::append(const char* name, bool value, const char* delim);
template BRAP_API void BParameters::append(const char* name, int value, const char* delim);
template BRAP_API void BParameters::append(const char* name, unsigned int value, const char* delim);
template BRAP_API void BParameters::append(const char* name, std::string value, const char* delim);
template BRAP_API void BParameters::append(const char* name, float value, const char* delim);
template BRAP_API void BParameters::append(const char* name, char value, const char* delim);
template BRAP_API void BParameters::append(const char* name, const char* value, const char* delim);


template BRAP_API void BParameters::add(const char* name, bool value);
template BRAP_API void BParameters::add(const char* name, int value);
template BRAP_API void BParameters::add(const char* name, unsigned int value);
template BRAP_API void BParameters::add(const char* name, std::string value);
template BRAP_API void BParameters::add(const char* name, float value);
template BRAP_API void BParameters::add(const char* name, double value);
template BRAP_API void BParameters::add(const char* name, char value);
template BRAP_API void BParameters::add(const char* name, const char* value);


template BRAP_API void BParameters::set(const char* name, bool value);
template BRAP_API void BParameters::set(const char* name, int value);
template BRAP_API void BParameters::set(const char* name, unsigned int value);
template BRAP_API void BParameters::set(const char* name, std::string value);
template BRAP_API void BParameters::set(const char* name, float value);
template BRAP_API void BParameters::set(const char* name, double value);
template BRAP_API void BParameters::set(const char* name, char value);
template BRAP_API void BParameters::set(const char* name, const char* value);


template BRAP_API bool         BParameters::get(const char* name) const ;
template BRAP_API int          BParameters::get(const char* name) const ;
template BRAP_API unsigned int BParameters::get(const char* name) const ;
template BRAP_API std::string  BParameters::get(const char* name) const ;
template BRAP_API float        BParameters::get(const char* name) const ;
template BRAP_API double       BParameters::get(const char* name) const ;
template BRAP_API char         BParameters::get(const char* name) const ;
//template BRAP_API const char*  BParameters::get(const char* name) const ;


template BRAP_API bool         BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API int          BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API unsigned int BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API std::string  BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API float        BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API double       BParameters::get(const char* name, const char* fallback) const ;
template BRAP_API char         BParameters::get(const char* name, const char* fallback) const ;

//template BRAP_API const char*  BParameters::get(const char* name, const char* fallback) const ;



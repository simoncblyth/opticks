#include "GItemIndex.hh"

#include "assert.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "jsonutil.hpp"


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GItemIndex::add(const char* name, unsigned int source)
{
    // only the first ocurrence of a repeated name is added
    if(m_source.count(name)==0)
    { 
        m_source[name] = source ;
        unsigned int local = m_local.size() + 1 ; // 1-based index in addition order  
        m_local[name] = local ; 

        m_source2local[source] = local ; 
        m_local2source[local]  = source ; 
    }
}

unsigned int GItemIndex::getIndexLocal(const char* name, unsigned int missing)
{
    return m_local.count(name) == 1 ? m_local[name] : missing ; 
}

unsigned int GItemIndex::getNumItems()
{
    assert(m_source.size() == m_local.size());
    return m_local.size();
}

bool GItemIndex::operator() (const std::string& a, const std::string& b)
{
    // sort order for dump 
    return m_local[a] < m_local[b] ; 
}

void GItemIndex::dump(const char* msg)
{
   LOG(info) << msg << " itemtype: " << m_itemtype  ; 

   typedef std::map<std::string, unsigned int> MSU ; 
   typedef std::vector<std::string> VS ; 

   VS keys ; 
   for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) keys.push_back(it->first) ;

   std::sort(keys.begin(), keys.end(), *this );

   for(VS::iterator it=keys.begin() ; it != keys.end() ; it++ )
   {
       std::string k = *it ; 
       std::cout 
            << " name   " << std::setw(25) <<  k
            << " source " << std::setw(10) <<  m_source[k]
            << " local  " << std::setw(10) <<  m_local[k]
            << std::endl ; 
   }
}

void GItemIndex::test(const char* msg)
{
   LOG(info) << msg << " itemtype: " << m_itemtype  ; 

   typedef std::map<std::string, unsigned int> MSU ; 
   typedef std::vector<std::string> VS ; 

   VS keys ; 
   for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) keys.push_back(it->first) ;

   std::sort(keys.begin(), keys.end(), *this );

   for(VS::iterator it=keys.begin() ; it != keys.end() ; it++ )
   {
       std::string k = *it ; 
       unsigned int local  = m_local[k];
       unsigned int source = m_source[k];

       assert(strcmp(getNameLocal(local),k.c_str())==0); 
       assert(strcmp(getNameSource(source),k.c_str())==0); 
       assert(getIndexLocal(k.c_str())==local); 
       assert(getIndexSource(k.c_str())==source); 
       assert(convertSourceToLocal(source)==local); 
       assert(convertLocalToSource(local)==source); 

       std::cout 
            << " name   " << std::setw(25) <<  k
            << " source " << std::setw(10) <<  source
            << " local  " << std::setw(10) <<  local
            << std::endl ; 
   }
}



void GItemIndex::crossreference()
{
   typedef std::map<std::string, unsigned int> MSU ; 
   typedef std::vector<std::string> VS ; 

   VS keys ; 
   for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) keys.push_back(it->first) ;

   std::sort(keys.begin(), keys.end(), *this );

   for(VS::iterator it=keys.begin() ; it != keys.end() ; it++ )
   {
       std::string k = *it ; 
       unsigned int source = m_source[k];
       unsigned int local  = m_local[k];

       m_source2local[source] = local ; 
       m_local2source[local]  = source ; 
   }
}


void GItemIndex::save(const char* idpath)
{
    saveMap<std::string, unsigned int>( m_source, idpath, getPrefixedString("Source.json").c_str() );  
    saveMap<std::string, unsigned int>( m_local , idpath, getPrefixedString("Local.json").c_str() );  
}

std::string GItemIndex::getPrefixedString(const char* tail)
{
    std::string prefix(m_itemtype); 
    return prefix + tail ; 
}

void GItemIndex::loadMaps(const char* idpath)
{
    loadMap<std::string, unsigned int>( m_source, idpath, getPrefixedString("Source.json").c_str() );  
    loadMap<std::string, unsigned int>( m_local , idpath, getPrefixedString("Local.json").c_str() );  
    crossreference();
}


unsigned int GItemIndex::getIndexSource(const char* name, unsigned int missing)
{
    return m_source.count(name) == 1 ? m_source[name] : missing ; 
}
const char* GItemIndex::getNameLocal(unsigned int local, const char* missing)
{
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) 
        if(it->second == local) return it->first.c_str();
    return missing ; 
}
const char* GItemIndex::getNameSource(unsigned int source, const char* missing)
{
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_source.begin() ; it != m_source.end() ; it++ ) 
        if(it->second == source) return it->first.c_str();
    return missing ; 
}


unsigned int GItemIndex::convertSourceToLocal(unsigned int source, unsigned int missing)
{
    return m_source2local.count(source) == 1 ? m_source2local[source] : missing ; 
}

unsigned int GItemIndex::convertLocalToSource(unsigned int local, unsigned int missing)
{
    return m_local2source.count(local) == 1 ? m_local2source[local] : missing ; 
}






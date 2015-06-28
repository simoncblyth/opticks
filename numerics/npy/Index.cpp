#include "Index.hpp"

#include "assert.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>

#include "jsonutil.hpp"


#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void Index::add(const char* name, unsigned int source)
{
    // only the first ocurrence of a repeated name is added
    if(m_source.count(name)==0)
    { 
        m_source[name] = source ;
        unsigned int local = m_local.size() + 1 ; // 1-based index in addition order  
        m_local[name] = local ; 

        m_source2local[source] = local ; 
        m_local2source[local]  = source ; 
   
        sortNames(); // when dealing with very big indices could just do this after all adds are made 
    }
}

void Index::sortNames()
{
   typedef std::map<std::string, unsigned int> MSU ; 
   m_names.clear();
   for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) m_names.push_back(it->first) ;
   std::sort(m_names.begin(), m_names.end(), *this ); // ascending local index
}


bool Index::operator() (const std::string& a, const std::string& b)
{
    // sort order for dump 
    return m_local[a] < m_local[b] ; 
}

unsigned int Index::getIndexLocal(const char* name, unsigned int missing)
{
    return m_local.count(name) == 1 ? m_local[name] : missing ; 
}

unsigned int Index::getNumItems()
{
    assert(m_source.size() == m_local.size());
    return m_local.size();
}


unsigned int Index::getIndexSource(const char* name, unsigned int missing)
{
    return m_source.count(name) == 1 ? m_source[name] : missing ; 
}
const char* Index::getNameLocal(unsigned int local, const char* missing)
{
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_local.begin() ; it != m_local.end() ; it++ ) 
        if(it->second == local) return it->first.c_str();
    return missing ; 
}
const char* Index::getNameSource(unsigned int source, const char* missing)
{
    typedef std::map<std::string, unsigned int> MSU ; 
    for(MSU::iterator it=m_source.begin() ; it != m_source.end() ; it++ ) 
        if(it->second == source) return it->first.c_str();
    return missing ; 
}


unsigned int Index::convertSourceToLocal(unsigned int source, unsigned int missing)
{
    return m_source2local.count(source) == 1 ? m_source2local[source] : missing ; 
}

unsigned int Index::convertLocalToSource(unsigned int local, unsigned int missing)
{
    return m_local2source.count(local) == 1 ? m_local2source[local] : missing ; 
}


void Index::test(const char* msg, bool verbose)
{
   LOG(info) << msg << " itemtype: " << m_itemtype  ; 

   typedef std::vector<std::string> VS ; 
   for(VS::iterator it=m_names.begin() ; it != m_names.end() ; it++ )
   {
       std::string iname = *it ; 
       unsigned int local  = m_local[iname];
       unsigned int source = m_source[iname];

       assert(strcmp(getNameLocal(local),iname.c_str())==0); 
       assert(strcmp(getNameSource(source),iname.c_str())==0); 
       assert(getIndexLocal(iname.c_str())==local); 
       assert(getIndexSource(iname.c_str())==source); 
       assert(convertSourceToLocal(source)==local); 
       assert(convertLocalToSource(local)==source); 

       if(verbose) std::cout 
            << " name   " << std::setw(35) <<  iname
            << " source " << std::setw(10) <<  std::dec << source
            << " local  " << std::setw(10) <<  std::dec << local
            << std::endl ; 

   }
}

void Index::dump(const char* msg)
{
    test(msg, true);
}



void Index::crossreference()
{
   typedef std::map<std::string, unsigned int> MSU ; 
   typedef std::vector<std::string> VS ; 


   for(VS::iterator it=m_names.begin() ; it != m_names.end() ; it++ )
   {
       std::string k = *it ; 
       unsigned int source = m_source[k];
       unsigned int local  = m_local[k];

       m_source2local[source] = local ; 
       m_local2source[local]  = source ; 
   }
}


void Index::save(const char* idpath)
{
    saveMap<std::string, unsigned int>( m_source, idpath, getPrefixedString("Source").c_str() );  
    saveMap<std::string, unsigned int>( m_local , idpath, getPrefixedString("Local").c_str() );  
}

std::string Index::getPrefixedString(const char* tail)
{
    std::string prefix(m_itemtype); 
    return prefix + tail + m_ext ; 
}

void Index::loadMaps(const char* idpath)
{
    loadMap<std::string, unsigned int>( m_source, idpath, getPrefixedString("Source").c_str() );  
    loadMap<std::string, unsigned int>( m_local , idpath, getPrefixedString("Local").c_str() );  

    sortNames();
    crossreference();
}


Index* Index::load(const char* idpath, const char* itemtype)
{
    Index* idx = new Index(itemtype);
    idx->loadMaps(idpath);
    return idx ; 
}



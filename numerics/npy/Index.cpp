#include "Index.hpp"

#include "assert.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>

#include "jsonutil.hpp"
#include "NLog.hpp"


std::string Index::description()
{
    std::stringstream ss ; 

    const char* type = getItemType() ;
    const char* title = getTitle() ;

    ss << "Index" 
       << " itemtype " << ( type ? type : "NULL" )
       << " title " << ( title ? title : "NULL" )
       << " numItems " << getNumItems()
       ;

    return ss.str();
}



void Index::add(const VS& vs)
{
    unsigned int n(0);
    for(VS::const_iterator it=vs.begin(); it!=vs.end() ; it++) 
    {
        add(it->c_str(),n+1);
        n++ ;
    }
}

void Index::add(const char* name, unsigned int source, bool sort )
{
    // only the first ocurrence of a repeated name is added
    // local index incremented for each unique name
    if(m_source.count(name)==0)
    { 
        m_source[name] = source ;

       // historically have been using : 1-based index in addition order  
        unsigned int local = m_onebased ? m_local.size() + 1 : m_local.size() ; 
        m_local[name] = local ; 

        m_source2local[source] = local ; 
        m_local2source[local]  = source ; 
   
        if(sort) sortNames(); // when dealing with very big indices could just do this after all adds are made 
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

bool Index::hasItem(const char* name)
{
    return m_local.count(name) == 1 ; 
}


unsigned int Index::getNumItems()
{
    //assert(m_source.size() == m_local.size());
    return m_local.size();
}


// fulfil NSequence
//   NSequence indices are zero-based, so have to convert when one-based
unsigned int Index::getNumKeys()
{
    return m_local.size();
}
const char* Index::getKey(unsigned int i)
{
    unsigned int local = m_onebased ? i + 1 : i ;
    return getNameLocal(local);
}
unsigned int Index::getIndex(const char* key)
{
    unsigned int local = getIndexLocal(key);
    return m_onebased ? local - 1 : local ; 
}


unsigned int Index::getIndexSourceTotal()
{
    if(m_source_total == 0)
    {
        typedef std::map<std::string, unsigned int> MSU ; 
        for(MSU::iterator it=m_source.begin() ; it != m_source.end() ; it++ ) m_source_total += it->second ; 
    }
    return m_source_total ; 
}

float Index::getIndexSourceFraction(const char* name)
{
     unsigned int total = getIndexSourceTotal();
     unsigned int value = getIndexSource(name);
     return total > 0 ? float(value)/float(total) : 0.f ; 
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

       //assert(strcmp(getNameSource(source),iname.c_str())==0); 
       if(strcmp(getNameSource(source),iname.c_str())!=0) 
           LOG(warning) << "Index::test inconsistency " 
                        << " source " << source 
                        << " iname " << iname 
                        ;

       assert(getIndexLocal(iname.c_str())==local); 
       assert(getIndexSource(iname.c_str())==source); 

       //assert(convertSourceToLocal(source)==local); 
       if(convertSourceToLocal(source)!=local)
           LOG(warning) << "Index::test convertSourceToLocal inconsistency " 
                        << " source " << source 
                        << " local " << local 
                        ;

       //assert(convertLocalToSource(local)==source); 
       if(convertLocalToSource(local)!=source) 
           LOG(warning) << "Index::test convertLocalToSource inconsistency " 
                        << " source " << source 
                        << " local " << local 
                        ;


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

std::string Index::directory(const char* pfold, const char* rfold)
{
    std::stringstream ss ; 
    ss << pfold << "/" << rfold ; 
    std::string dir = ss.str();
    return dir ; 
}

void Index::save(const char* pfold, const char* rfold)
{
   std::string dir = directory(pfold, rfold);
   save(dir.c_str());
}

void Index::save(const char* idpath)
{
    std::string sname = getPrefixedString("Source") ;
    std::string lname = getPrefixedString("Local") ;

    LOG(info) << "Index::save"
              << " sname " << sname 
              << " lname " << lname 
              << " itemtype " << m_itemtype
              << " ext " << m_ext 
              ;

    saveMap<std::string, unsigned int>( m_source, idpath, sname.c_str() );  
    saveMap<std::string, unsigned int>( m_local , idpath, lname.c_str() );  
}
std::string Index::getPrefixedString(const char* tail)
{
    std::string prefix(m_itemtype); 
    return prefix + tail + m_ext ; 
}

std::string Index::getPath(const char* idpath, const char* prefix)
{
     bool create_idpath_dir = true ; 
     std::string path = preparePath(idpath, getPrefixedString(prefix).c_str(), create_idpath_dir);
     return path;
}



bool Index::exists(const char* idpath)
{
    bool sx = existsPath(idpath, getPrefixedString("Source").c_str());
    bool lx = existsPath(idpath, getPrefixedString("Local").c_str());
    return sx && lx ; 
}

void Index::loadMaps(const char* idpath)
{
    loadMap<std::string, unsigned int>( m_source, idpath, getPrefixedString("Source").c_str() );  
    loadMap<std::string, unsigned int>( m_local , idpath, getPrefixedString("Local").c_str() );  

    sortNames();
    crossreference();
}

Index* Index::load(const char* pfold, const char* rfold, const char* itemtype)
{
   std::string dir = directory(pfold, rfold);
   return load(dir.c_str(), itemtype );
}

Index* Index::load(const char* idpath, const char* itemtype)
{
    Index* idx = new Index(itemtype);
    if(idx->exists(idpath))
    {
       idx->loadMaps(idpath);
    }
    else
    {
        LOG(warning) << "Index::load FAILED to load index " 
                     << " idpath " << idpath 
                     << " itemtype " << itemtype 
                     << " Source path " << idx->getPath(idpath, "Source")
                     << " Local path " << idx->getPath(idpath, "Local")
                     ;
        idx = NULL ;
    }
    return idx ; 
}

void Index::dumpPaths(const char* idpath, const char* msg)
{

    bool sx = existsPath(idpath, getPrefixedString("Source").c_str());
    bool lx = existsPath(idpath, getPrefixedString("Local").c_str());

    LOG(info) << msg 
              << std::endl
              << " Source:" << ( sx ? "EXISTS " : "MISSING" ) << getPath(idpath, "Source")
              << "  Local:" << ( lx ? "EXISTS " : "MISSING" ) << getPath(idpath, "Local")
              ;
}




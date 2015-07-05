#include "GItemIndex.hh"
#include "GColorMap.hh"
#include "GColors.hh"
#include "GBuffer.hh"

#include "assert.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>

#include "Index.hpp"
#include "jsonutil.hpp"


#ifdef GUI_
#include <imgui.h>
#endif

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


GItemIndex* GItemIndex::load(const char* idpath, const char* itemtype)
{
    GItemIndex* idx = new GItemIndex(itemtype) ;    // itemname->index
    idx->loadIndex(idpath);
    return idx ; 
}

void GItemIndex::init(const char* itemtype)
{
    m_index = new Index(itemtype);
}

void GItemIndex::loadIndex(const char* idpath, const char* override)
{
    const char* itemtype = override ? override : m_index->getItemType() ;
    if(override)
    {
        LOG(warning)<<"GItemIndex::loadIndex using override itemtype " << itemtype << " instead of default " << m_index->getItemType() ;
    }
    m_index = Index::load(idpath, itemtype);
}

void GItemIndex::add(const char* name, unsigned int source)
{
    m_index->add(name, source);
}

unsigned int GItemIndex::getIndexLocal(const char* name, unsigned int missing)
{
    return m_index->getIndexLocal(name, missing);
}

//const char* GItemIndex::getItemType()
//{
//    return m_index->getItemType() ; 
//}

//std::vector<std::string>& GItemIndex::getNames()
//{
//    return m_index->getNames() ;  
//}

void GItemIndex::save(const char* idpath)
{
    m_index->save(idpath);
}

unsigned int GItemIndex::getNumItems()
{
    return m_index->getNumItems();
}

void GItemIndex::test(const char* msg, bool verbose)
{
    m_index->test(msg, verbose);
}

void GItemIndex::dump(const char* msg)
{
   LOG(info) << msg << " itemtype: " << m_index->getItemType()  ; 

   typedef std::vector<std::string> VS ; 
   VS names = m_index->getNames();
   for(VS::iterator it=names.begin() ; it != names.end() ; it++ )
   {
       std::string iname = *it ; 
       const char*  cname = m_colormap ? m_colormap->getItemColor(iname.c_str(), NULL) : NULL ; 
       unsigned int ccode = m_colors   ? m_colors->getCode(cname, 0xFFFFFF) : 0xFFFFFF ; 

       unsigned int source = m_index->getIndexSource(iname.c_str()) ;
       unsigned int local  = m_index->getIndexLocal(iname.c_str()) ;
       std::cout 
            << " iname  " << std::setw(35) <<  iname
            << " source " << std::setw(4) <<  std::dec << source
            << " local  " << std::setw(4) <<  std::dec << local
            << " 0x " << std::setw(4)     <<  std::hex << local
            << " cname  " << std::setw(20) <<  ( cname ? cname : "no-colormap-or-missing" )
            << " ccode  " << std::setw(20) << std::hex <<  ccode
            << std::endl ; 
   }
}



void GItemIndex::formTable()
{
   m_codes.clear(); 
   m_labels.clear(); 

   // collect keys (item names) into vector and sort into ascending local index order 

   typedef std::vector<std::string> VS ; 

   VS& names = m_index->getNames(); 
   for(VS::iterator it=names.begin() ; it != names.end() ; it++ )
   {
       std::string iname = *it ; 
       const char*  cname = m_colormap ? m_colormap->getItemColor(iname.c_str(), NULL) : NULL ; 
       unsigned int code  = m_colors   ? m_colors->getCode(cname, 0xFFFFFF) : 0xFFFFFF ; 
       unsigned int local  = m_index->getIndexLocal(iname.c_str()) ;

       std::stringstream ss ; 
       ss  << std::setw(5)  << std::dec << local 
           << std::setw(25) << iname
           << std::setw(25) << cname 
           << std::setw(10) << std::hex << code 
           ;

       m_codes.push_back(code);
       m_labels.push_back(ss.str());
   }
}


GBuffer* GItemIndex::makeColorBuffer()
{
   if(m_colors==NULL)
       LOG(warning) << "GItemIndex::makeColorBuffer no colors defined will provide defaults"  ; 

   formTable(); 
   LOG(info) << "GItemIndex::makeColorBuffer codes " << m_codes.size() ;  
   return m_colors->make_uchar4_buffer(m_codes) ; 
}

GBuffer* GItemIndex::getColorBuffer()
{
   if(!m_colorbuffer)
   {
       m_colorbuffer = makeColorBuffer();
   }  
   return m_colorbuffer ; 
}


void GItemIndex::gui()
{
#ifdef GUI_    
    if (ImGui::CollapsingHeader(m_index->getItemType()))
    {
       for(unsigned int i=0 ; i < m_labels.size() ; i++)
       {
           unsigned int code = m_codes[i] ;
           unsigned int red   = (code & 0xFF0000) >> 16 ;
           unsigned int green = (code & 0x00FF00) >>  8 ;
           unsigned int blue  = (code & 0x0000FF)  ;
           ImGui::TextColored(ImVec4(red/256.,green/256.,blue/256.,1.0f), m_labels[i].c_str() );
       }
    }  
#endif
}




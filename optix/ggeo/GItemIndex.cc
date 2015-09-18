#include "GItemIndex.hh"
#include "GColorMap.hh"
#include "GColors.hh"
#include "GBuffer.hh"
#include "GVector.hh"

#include "assert.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>

// npy-
#include "Index.hpp"
#include "Types.hpp"
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

void GItemIndex::setTitle(const char* title)
{
   m_index->setTitle(title);
}

int GItemIndex::getSelected()
{
   return m_index->getSelected();
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

bool GItemIndex::hasItem(const char* key)
{
    return m_index->hasItem(key);
}


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
            << std::dec
            << std::endl ; 
   }
}





std::string GItemIndex::getLabel(const char* key, unsigned int& colorcode)
{
    return (*m_labeller)(this, key, colorcode);
}



void GItemIndex::setLabeller(Labeller_t labeller )
{
    switch(labeller)
    {
        case    DEFAULT  : setLabeller(&GItemIndex::defaultLabeller)     ; break ; 
        case   COLORKEY  : setLabeller(&GItemIndex::colorKeyLabeller)    ; break ; 
        case MATERIALSEQ : setLabeller(&GItemIndex::materialSeqLabeller) ; break ; 
        case  HISTORYSEQ : setLabeller(&GItemIndex::historySeqLabeller)  ; break ; 
    }
}

std::string GItemIndex::defaultLabeller(GItemIndex* self, const char* key, unsigned int& colorcode)
{
   colorcode = 0xFFFFFF ; 
   return key ;  
}

std::string GItemIndex::colorKeyLabeller(GItemIndex* self, const char* key, unsigned int& colorcode )
{
    // function pointers have to be static, so access members python style
    colorcode = self->getColorCode(key);

    Index* index = self->getIndex();
    unsigned int local  = index->getIndexLocal(key) ;

    std::stringstream ss ; 
    ss  << std::setw(5)  << std::dec << local 
        << std::setw(25) << key
        << std::setw(10) << std::hex << colorcode 
        ;

    return ss.str();
}


const char* GItemIndex::getColorName(const char* key)
{
    return m_colormap ? m_colormap->getItemColor(key, NULL) : NULL ; 
}

unsigned int GItemIndex::getColorCode(const char* key )
{
    const char*  colorname =  getColorName(key) ;
    unsigned int colorcode  = m_colors ? m_colors->getCode(colorname, 0xFFFFFF) : 0xFFFFFF ; 
    return colorcode ; 
}


gfloat3* GItemIndex::makeColor( unsigned int rgb )
{
    unsigned int red   =  ( rgb & 0xFF0000 ) >> 16 ;  
    unsigned int green =  ( rgb & 0x00FF00 ) >>  8 ;  
    unsigned int blue  =  ( rgb & 0x0000FF ) ;  

    float d(0xFF);
    float r = float(red)/d ;
    float g = float(green)/d ;
    float b = float(blue)/d ;

    return new gfloat3( r, g, b) ;
}



std::string GItemIndex::materialSeqLabeller(GItemIndex* self, const char* key_, unsigned int& colorcode)
{
   colorcode = 0xFFFFFF ; 
   std::string key(key_);
   Types* types = self->getTypes();
   assert(types); 

   std::string seqmat = types->abbreviateHexSequenceString(key, Types::MATERIALSEQ);  

   return seqmat ;  
}

std::string GItemIndex::historySeqLabeller(GItemIndex* self, const char* key_, unsigned int& colorcode)
{
   colorcode = 0xFFFFFF ; 
   std::string key(key_);

   Types* types = self->getTypes();
   assert(types); 
   std::string seqhis = types->abbreviateHexSequenceString(key, Types::HISTORYSEQ);  
   return seqhis ;  
}



void GItemIndex::formTable(bool verbose)
{
   m_codes.clear(); 
   m_labels.clear(); 

   // collect keys (item names) into vector and sort into ascending local index order 

   typedef std::vector<std::string> VS ; 

   if(verbose) LOG(info)<<"GItemIndex::formTable " ;

   VS& names = m_index->getNames(); 
   for(VS::iterator it=names.begin() ; it != names.end() ; it++ )
   {
       std::string key = *it ; 
       unsigned int colorcode(0x0) ; 
       std::string label = getLabel(key.c_str(), colorcode );

       const char* colorname = getColorName(key.c_str());
       if(colorname)
       {
          label += "   " ; 
          label += colorname ; 
       }

       if(verbose) 
           std::cout
            << std::setw(30) << key 
            << " : " 
            << label 
            << std::endl; 

       m_codes.push_back(colorcode);
       m_labels.push_back(label);
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

void GItemIndex::gui_radio_select()
{
#ifdef GUI_
    if (ImGui::CollapsingHeader(m_index->getTitle()))
    {
       typedef std::vector<std::string> VS ; 
       VS names = m_index->getNames();
       assert(names.size() == m_labels.size());

       int* ptr = m_index->getSelectedPtr();

       std::string all("All ");
       all += m_index->getItemType() ;  

       ImGui::RadioButton( all.c_str(), ptr, 0 );

       for(unsigned int i=0 ; i < m_labels.size() ; i++)
       {
           std::string iname = names[i] ;
           std::string label = m_labels[i] ;
           unsigned int local  = m_index->getIndexLocal(iname.c_str()) ;
           ImGui::RadioButton( label.c_str(), ptr, local);
       }   
       ImGui::Text("%s %d ", m_index->getItemType(), *ptr);
   }
#endif
}

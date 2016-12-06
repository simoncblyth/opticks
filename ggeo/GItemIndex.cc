#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sstream>


// okc-
#include "OpticksAttrSeq.hh"
#include "OpticksColors.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "Index.hpp"
#include "Types.hpp"

#include "NQuad.hpp"


#include "GColorMap.hh"
#include "GVector.hh"
#include "GItemIndex.hh"

#include "PLOG.hh"


GItemIndex* GItemIndex::load(const char* idpath, const char* itemtype)
{
    GItemIndex* idx = new GItemIndex(itemtype) ;    // itemname->index
    idx->loadIndex(idpath);
    return idx ; 
}

GItemIndex::GItemIndex(const char* itemtype)
   : 
   m_index(NULL),
   m_colors(NULL),
   m_colormap(NULL),
   m_colorbuffer(NULL),
   m_types(NULL),
   m_handler(NULL)
{
   init(itemtype);
   setLabeller(DEFAULT);
}

GItemIndex::GItemIndex(Index* index)
   : 
   m_index(index),
   m_colors(NULL),
   m_colormap(NULL),
   m_colorbuffer(NULL),
   m_types(NULL),
   m_handler(NULL)
{
   setLabeller(DEFAULT);
}


void GItemIndex::setLabeller(GItemIndexLabellerPtr labeller)
{
   m_labeller = labeller ; 
}
void GItemIndex::setColorSource(OpticksColors* colors)
{
   m_colors = colors ; 
}
void GItemIndex::setColorMap(GColorMap* colormap)
{
   m_colormap = colormap ; 
}
void GItemIndex::setTypes(Types* types)
{
   m_types = types ; 
}
void GItemIndex::setHandler(OpticksAttrSeq* handler)
{
   m_handler = handler ; 
}



Types* GItemIndex::getTypes()
{
   return m_types ;
}


OpticksColors* GItemIndex::getColorSource()
{
   return m_colors ; 
}
GColorMap* GItemIndex::getColorMap()
{
   return m_colormap ; 
}

Index* GItemIndex::getIndex()
{
   return m_index ; 
}


std::vector<unsigned int>& GItemIndex::getCodes()
{
   return m_codes ; 
}

std::vector<std::string>& GItemIndex::getLabels()
{
   return m_labels ; 
}

const char* GItemIndex::getLabel(unsigned index)
{
   return index < m_labels.size() ? m_labels[index].c_str() : NULL ;
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
const char* GItemIndex::getSelectedKey()
{
   return m_index->getSelectedKey();
}
const char* GItemIndex::getSelectedLabel()
{
   int sel = m_index->getSelected();
   return sel == 0 ? "All" : getLabel(sel-1);
}






void GItemIndex::loadIndex(const char* idpath, const char* override)
{
    const char* itemtype = override ? override : m_index->getItemType() ;
    if(override)
    {
        LOG(warning)<<"GItemIndex::loadIndex using override itemtype " << itemtype << " instead of default " << m_index->getItemType() ;
    }
    m_index = Index::load(idpath, itemtype);
    if(!m_index)
        LOG(warning) << "GItemIndex::loadIndex"
                     << " failed for "
                     << " idpath " << idpath
                     << " override " << ( override ? override : "NULL" )
                     ; 

}

void GItemIndex::add(const char* name, unsigned int source)
{
    assert(m_index);
    m_index->add(name, source);
}

unsigned int GItemIndex::getIndexLocal(const char* name, unsigned int missing)
{
    assert(m_index);
    return m_index->getIndexLocal(name, missing);
}

bool GItemIndex::hasItem(const char* key)
{
    assert(m_index);
    return m_index->hasItem(key);
}


void GItemIndex::save(const char* idpath)
{
    assert(m_index);
    m_index->save(idpath);
}

unsigned int GItemIndex::getNumItems()
{
    assert(m_index);
    return m_index->getNumItems();
}

void GItemIndex::test(const char* msg, bool verbose)
{
    assert(m_index);
    m_index->test(msg, verbose);
}

void GItemIndex::dump(const char* msg)
{
   if(!m_index)
   {
       LOG(warning) << msg << " NULL index "; 
       return ; 
   }  

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
    // Trojan Handler : out to kill most of this class
    return m_handler ? 
                       m_handler->getLabel(m_index, key, colorcode) 
                     : 
                      (*m_labeller)(this, key, colorcode);
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

std::string GItemIndex::defaultLabeller(GItemIndex* /*self*/, const char* key, unsigned int& colorcode)
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


nvec3 GItemIndex::makeColor( unsigned int rgb )
{
    unsigned int red   =  ( rgb & 0xFF0000 ) >> 16 ;  
    unsigned int green =  ( rgb & 0x00FF00 ) >>  8 ;  
    unsigned int blue  =  ( rgb & 0x0000FF ) ;  

    float d(0xFF);
    float r = float(red)/d ;
    float g = float(green)/d ;
    float b = float(blue)/d ;

    return make_nvec3( r, g, b) ;
}



std::string GItemIndex::materialSeqLabeller(GItemIndex* self, const char* key_, unsigned int& colorcode)
{
   colorcode = 0xFFFFFF ; 
   std::string key(key_);
   Types* types = self->getTypes();
   assert(types); 

   std::string seqmat = types->abbreviateHexSequenceString(key, Types::MATERIALSEQ);  
   std::stringstream ss ; 
   ss << std::setw(16) << key_ 
      << " "
      << seqmat 
      ;

   return ss.str() ;  
}

std::string GItemIndex::historySeqLabeller(GItemIndex* self, const char* key_, unsigned int& colorcode)
{
   colorcode = 0xFFFFFF ; 
   std::string key(key_);

   Types* types = self->getTypes();
   assert(types); 
   std::string seqhis = types->abbreviateHexSequenceString(key, Types::HISTORYSEQ);  

   std::stringstream ss ; 
   ss << std::setw(16) << key_ 
      << " "
      << seqhis
      ;

   return ss.str() ;  
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






NPY<unsigned char>* GItemIndex::makeColorBuffer()
{
   if(m_colors==NULL)
       LOG(warning) << "GItemIndex::makeColorBuffer no colors defined will provide defaults"  ; 

   formTable(); 
   LOG(info) << "GItemIndex::makeColorBuffer codes " << m_codes.size() ;  
   return m_colors->make_buffer(m_codes) ; 
}

NPY<unsigned char>* GItemIndex::getColorBuffer()
{
   if(!m_colorbuffer)
   {
       m_colorbuffer = makeColorBuffer();
   }  
   return m_colorbuffer ; 
}



std::string GItemIndex::gui_radio_select_debug()
{
    assert(m_index);
    Index* ii = m_index ; 

    std::stringstream ss ; 
    typedef std::vector<std::string> VS ;

    VS  names = ii->getNames();
    VS& labels = getLabels(); 

    LOG(info) << "GItemIndex::gui_radio_select_debug"
              << " names " << names.size()
              << " labels " << labels.size()
              ;

    assert(names.size() == labels.size());

    ss << " title " << ii->getTitle() << std::endl ;
    for(unsigned int i=0 ; i < labels.size() ; i++) 
            ss << std::setw(3) << i 
               << " name  " << std::setw(20) << names[i]
               << " label " << std::setw(50) << labels[i]
               << std::endl ; 

    return ss.str();
}



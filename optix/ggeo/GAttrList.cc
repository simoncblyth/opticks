#include "GAttrList.hh"

#include "GItemList.hh"
#include "GCache.hh"
#include "GColors.hh"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



const char* GAttrList::getColorName(const char* key)
{
    return m_color.count(key) == 1 ? m_color[key].c_str() : NULL ; 
}

unsigned int GAttrList::getColorCode(const char* key )
{
    const char*  colorname =  getColorName(key) ;
    GColors* palette = m_cache->getColors();
    unsigned int colorcode  = palette->getCode(colorname, 0xFFFFFF) ; 
    return colorcode ; 
}

std::vector<unsigned int>& GAttrList::getColorCodes()
{
    if(m_names && m_color_codes.size() == 0)
    {
        unsigned int ni = m_names->getNumItems();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            std::string& item = m_names->getItem(i);
            assert(!item.empty()); 
            unsigned int code = getColorCode(item.c_str());
            m_color_codes.push_back(code);
        }         
    }
    return m_color_codes ; 
}

std::vector<std::string>& GAttrList::getLabels()
{
    if(m_names && m_labels.size() == 0)
    {
        unsigned int ni = m_names->getNumItems();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            std::string& item = m_names->getItem(i);
            const char*  colorname = getColorName(item.c_str()) ;
            unsigned int colorcode = getColorCode(item.c_str()) ;

            std::stringstream ss ;    

            // default label 
            ss << std::setw(3)  << i 
               << std::setw(30) << item 
               << std::setw(10) << std::hex << colorcode << std::dec
               << std::setw(15) << ( colorname ? colorname : "" )
               ;

            m_labels.push_back(ss.str());
        }     
    }
    return m_labels ; 
}


std::string GAttrList::getAbbr(const char* shortname)
{
    return m_abbrev.count(shortname) == 1 ? m_abbrev[shortname] : shortname ; 
}


void GAttrList::dump(const char* names, const char* msg)
{
    if(!m_names) return ; 

    typedef std::vector<std::string> VS ; 
    VS elem ; 
    boost::split(elem, names, boost::is_any_of(","));

    LOG(info) << msg << " " << names ; 
    for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++)
    {
        const char* key = it->c_str();
        unsigned int idx = m_names->getIndex(key);
        if(idx == GItemList::UNSET)
        {
             LOG(warning) << "GAttrList::dump no item named: " << *it ; 
        }
        else
        {
             std::string abbr = getAbbr(key);  
             const char* colorname = getColorName(key);  
             unsigned int colorcode = getColorCode(key);              

             std::cout << std::setw(5) << idx 
                       << std::setw(30) << *it 
                       << std::setw(10) << std::hex << colorcode << std::dec
                       << std::setw(15) << colorname 
                       << std::setw(15) << abbr
                       << std::endl ; 
        }
    }
}




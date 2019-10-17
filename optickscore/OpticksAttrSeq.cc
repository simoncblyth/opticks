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


#include <cstring>
#include <climits>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <boost/lexical_cast.hpp>
//#include <boost/algorithm/string.hpp>
#include "SLog.hh"

// brap-
#include "BStr.hh"
#include "BHex.hh"
#include "PLOG.hh"

// npy-
#include "NMeta.hpp"
#include "Index.hpp"

//opticks-
#include "OpticksAttrSeq.hh"
#include "Opticks.hh"
#include "OpticksResource.hh"
#include "OpticksColors.hh"

#include "PLOG.hh"

unsigned int OpticksAttrSeq::UNSET = UINT_MAX ; 
unsigned int OpticksAttrSeq::ERROR_COLOR = 0xAAAAAA ; 

const plog::Severity OpticksAttrSeq::LEVEL = PLOG::EnvLevel("OpticksAttrSeq", "debug"); 


OpticksAttrSeq::OpticksAttrSeq(Opticks* ok, const char* type)
   :
   m_log(new SLog("OpticksAttrSeq::OpticksAttrSeq","",verbose)),
   m_ok(ok),
   m_resource(m_ok->getResource()),
   m_type(strdup(type)),
   m_ctrl(0),
   m_sequence(NULL),
   m_abbrev_meta(NULL)
{
   init();
   (*m_log)("DONE");
}

const char* OpticksAttrSeq::getType()
{
    return m_type ; 
}

std::map<std::string, unsigned int>&  OpticksAttrSeq::getOrder()
{
    return m_order ;
}

void OpticksAttrSeq::setCtrl(unsigned char ctrl)
{
    m_ctrl = ctrl ; 
}

void OpticksAttrSeq::setAbbrevMeta(NMeta* abbrev)
{
    m_abbrev_meta = abbrev ; 
}



bool OpticksAttrSeq::hasSequence()
{
    return m_sequence != NULL ; 
}


void OpticksAttrSeq::init()
{
    LOG(LEVEL) ; 
}

/**
OpticksAttrSeq::loadPrefs
----------------------------

Loads .json preferences files into the three map members : 

* m_color
* m_abbrev
* m_order

Different prefs for each type : GMaterialLib, GSurfaceLib etc..
can be used.

**/


void OpticksAttrSeq::loadPrefs()
{
    LOG(LEVEL) << "["  ; 
    if(m_resource->loadPreference(m_color, m_type, "color.json"))
    {
        LOG(LEVEL) << "color " << m_type ;
    }

    if(m_resource->loadPreference(m_abbrev, m_type, "abbrev.json"))
    {
        LOG(LEVEL) << "abbrev " << m_type ;
    }

    if(m_resource->loadPreference(m_order, m_type, "order.json"))
    {
        LOG(LEVEL) << "order " << m_type ;
    }
    LOG(LEVEL) << "]"  ; 
}

void OpticksAttrSeq::setSequence(NSequence* seq)
{
    m_sequence = seq ; 
}

std::map<unsigned int, std::string> OpticksAttrSeq::getNamesMap(unsigned char ctrl)
{
     std::map<unsigned int, std::string> mus ; 
     unsigned int ni = m_sequence->getNumKeys();
     for(unsigned int i=0 ; i < ni ; i++)
     {
         const char* key = m_sequence->getKey(i);
         unsigned int idx = ctrl & ONEBASED ? i + 1 : i ; 
         mus[idx] = key ; 
     }
     return mus ; 
}

const char* OpticksAttrSeq::getColorName(const char* key)
{
    return m_color.count(key) == 1 ? m_color[key].c_str() : NULL ; 
}

unsigned int OpticksAttrSeq::getColorCode(const char* key )
{
    const char*  colorname =  getColorName(key) ;
    OpticksColors* palette = m_ok->getColors();
    unsigned int colorcode  = palette->getCode(colorname, 0xFFFFFF) ; 
    return colorcode ; 
}

std::vector<unsigned int>& OpticksAttrSeq::getColorCodes()
{
    if(m_sequence && m_color_codes.size() == 0)
    {
        unsigned int ni = m_sequence->getNumKeys();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            const char* key = m_sequence->getKey(i);
            unsigned int code = key ? getColorCode(key) : ERROR_COLOR ;
            m_color_codes.push_back(code);
        }         
    }
    return m_color_codes ; 
}

std::vector<std::string>& OpticksAttrSeq::getLabels()
{
    if(m_sequence && m_labels.size() == 0)
    {
        unsigned int ni = m_sequence->getNumKeys();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            const char* key = m_sequence->getKey(i);
            const char*  colorname = key ? getColorName(key) : NULL ;
            unsigned int colorcode = key ? getColorCode(key) : ERROR_COLOR ;

            std::stringstream ss ;    

            // default label 
            ss << std::setw(3)  << i 
               << std::setw(30) << ( key ? key : "NULL" )
               << std::setw(10) << std::hex << colorcode << std::dec
               << std::setw(15) << ( colorname ? colorname : "" )
               ;

            m_labels.push_back(ss.str());
        }     
    }
    return m_labels ; 
}

/**
OpticksAttrSeq::getAbbr
--------------------------

See notes/issues/photon-flag-sequence-selection-history-flags-not-being-abbreviated-in-gui.rst

**/


std::string OpticksAttrSeq::getAbbr(const char* key)
{
    if(key == NULL) return "NULL" ; 

    std::string abb = key ; 
    if( m_abbrev_meta )
    {
        abb = m_abbrev_meta->get<std::string>(key, key ) ; 
    } 
    else
    {
        abb = m_abbrev.count(key) == 1 ? m_abbrev[key] : key ;  // copying key into string
    }
    return abb ; 
}


void OpticksAttrSeq::dump(const char* keys, const char* msg)
{

    if(!m_sequence) 
    {
        LOG(warning) << "OpticksAttrSeq::dump no sequence " ; 
        return ; 
    }
    LOG(info) << msg << " " << ( keys ? keys : "-" ) ; 

    if(keys)
    {
        typedef std::vector<std::string> VS ; 
        VS elem ; 
        BStr::split(elem, keys, ',');

        for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++)
             dumpKey(it->c_str()); 
    }
    else
    {
        unsigned int ni = m_sequence->getNumKeys();
        for(unsigned int i=0 ; i < ni ; i++)
        {
            const char* key = m_sequence->getKey(i);
            dumpKey(key);
        }
    }
}

void OpticksAttrSeq::dumpKey(const char* key)
{
    if(key == NULL)
    {
        LOG(warning) << "OpticksAttrSeq::dump NULL key " ;
        return ;   
    }
 

    unsigned int idx = m_sequence->getIndex(key);
    if(idx == UNSET)
    {
        LOG(warning) << "OpticksAttrSeq::dump no item named: " << key ; 
    }
    else
    {
        std::string abbr = getAbbr(key);  
        const char* colorname = getColorName(key);  
        unsigned int colorcode = getColorCode(key);              

        std::cout << " idx " << std::setw(5) << idx 
                  << " key " << std::setw(30) << key 
                  << " colorcode " << std::setw(10) << std::hex << colorcode << std::dec
                  << " colorname " << std::setw(15) << ( colorname ? colorname : "-" ) 
                  << " abbr " << std::setw(15) << abbr
                  << std::endl ; 
    }
}


/**
OpticksAttrSeq::decodeHexSequenceString
-----------------------------------------

Decodes hex keys like "4ccc1"  eg those from seqmat and seqhis

**/

std::string OpticksAttrSeq::decodeHexSequenceString(const char* seq, unsigned char ctrl)
{
    if(!seq) return "NULL" ;

    std::string lseq(seq);
    if(ctrl & REVERSE)
        std::reverse(lseq.begin(), lseq.end());
  
    std::stringstream ss ; 
    for(unsigned int i=0 ; i < lseq.size() ; i++) 
    {
        std::string sub = lseq.substr(i, 1) ;
        unsigned int local = BHex<unsigned int>::hex_lexical_cast(sub.c_str());
        unsigned int idx =  ctrl & ONEBASED ? local - 1 : local ;  
        const char* key = m_sequence->getKey(idx) ;
        std::string elem = ( ctrl & ABBREVIATE ) ? getAbbr(key) : key ; 
        ss << elem << " " ; 
    }
    return ss.str();
}



std::string OpticksAttrSeq::decodeString(const char* seq, unsigned char ctrl)
{
    // decodes keys like "-38"  eg those from boundaries
    if(!seq) return "NULL" ;

    std::stringstream ss ; 
    int code = boost::lexical_cast<int>(seq);
    unsigned int acode = abs(code);
    unsigned int idx = ctrl & ONEBASED ? acode - 1 : acode ;  
    const char* key = m_sequence->getKey(idx) ;
    std::string name = ( ctrl & ABBREVIATE ) ? getAbbr(key) : key ; 
    ss << name ; 
    return ss.str();
}



std::string OpticksAttrSeq::getLabel(Index* index, const char* key, unsigned int& colorcode)
{
    colorcode = 0xFFFFFF ;

    unsigned int source = index->getIndexSource(key); // the count for seqmat, seqhis
    float fraction      = index->getIndexSourceFraction(key);
    std::string dseq    = m_ctrl & HEXKEY 
                                ? 
                                   decodeHexSequenceString(key)
                                :
                                   decodeString(key)
                                ;

    std::stringstream ss ;  
    ss
        << std::setw(10) << source 
        << std::setw(10) << std::setprecision(3) << std::fixed << fraction 
        << std::setw(18) << ( key ? key : "-" ) 
        << " "  
        << std::setw(50) << dseq 
        ; 
    
    return ss.str();
}


void OpticksAttrSeq::dumpTable(Index* seqtab, const char* msg)
{
    LOG(info) << msg ;

    unsigned int colorcode(0xFFFFFF);

    std::stringstream ss ; 

    for(unsigned int i=0 ; i < seqtab->getNumKeys() ; i++)
    {
        const char* key = seqtab->getKey(i);
        std::string label = getLabel(seqtab, key, colorcode ); 
        ss << std::setw(5) << i
                  << label 
                  << std::endl ; 
    }

    ss << std::setw(5) << "TOT" 
              << std::setw(10) << seqtab->getIndexSourceTotal()
              << std::endl ;

    std::cout << ss.str() ;
}





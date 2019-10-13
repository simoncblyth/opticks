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

#pragma once

/*
History 

   GItemIndex 
        became overcomplicated to compensate for lack of persistable GMaterialLib, GSurfaceLib
        (GBoundaryLib attempted to skip a few steps and live without these)

   GItemList
        simple list of strings, used by GPropertyLib and subclasses 

   GAttrSeq
        a list with attributes like colors, abbreviations 

        provide the singing and dancing add-ons around the persistable GItemList 
        (by pulling from GItemIndex) without persistency 


   OpticksAttrSeq
         migration of ggeo-/GAttrSeq into optickscore- in move to get
         infrastructure out of ggeo- for wider access

*/

#include <map>
#include <string>
#include <vector>
#include "plog/Severity.h"

class SLog ; 

class Opticks ;
class OpticksResource ;
class NSequence ; 
class Index ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksAttrSeq {
        static const plog::Severity LEVEL ; 
    public:
        static unsigned int UNSET ; 
        static unsigned int ERROR_COLOR ; 

        enum {
                REVERSE    = 0x1 << 0,  
                ABBREVIATE = 0x1 << 1,
                ONEBASED   = 0x1 << 2,
                HEXKEY     = 0x1 << 3
             };

        enum {
                 SEQUENCE_DEFAULTS = REVERSE|ABBREVIATE|ONEBASED|HEXKEY,
                 VALUE_DEFAULTS = ABBREVIATE|ONEBASED
             };

    public:
        OpticksAttrSeq(Opticks* ok, const char* type);
        void setCtrl(unsigned char ctrl);
        void loadPrefs();
        const char* getType();
        std::map<std::string, unsigned int>& getOrder();
        void setSequence(NSequence* seq);
        bool hasSequence();
    public:
        //std::string  getCtrlDesc() const ; 
        std::string  getLabel(Index* index, const char* key, unsigned int& colorcode);
        std::string  getAbbr(const char* key);
        unsigned int getColorCode(const char* key );
        const char*  getColorName(const char* key);
        std::vector<unsigned int>& getColorCodes();
        std::vector<std::string>&  getLabels();
    public:
        std::map<unsigned int, std::string> getNamesMap(unsigned char ctrl=ONEBASED);
    public:
        std::string decodeHexSequenceString(const char* seq, unsigned char ctrl=REVERSE|ABBREVIATE|ONEBASED );
        std::string decodeString(const char* seq, unsigned char ctrl=ABBREVIATE|ONEBASED);
    public:
        void dump(const char* keys=NULL, const char* msg="OpticksAttrSeq::dump");
        void dumpKey(const char* key);
    public:
        void dumpTable(   Index* table, const char* msg="OpticksAttrSeq::dumpTable");
    private:
        void init();
    private:
        SLog*                                m_log ; 
        Opticks*                             m_ok ; 
        OpticksResource*                     m_resource ; 
        const char*                          m_type ; 
        unsigned char                        m_ctrl ; 
        NSequence*                           m_sequence ; 
    private:
        std::map<std::string, std::string>   m_abbrev ;
        std::map<std::string, std::string>   m_color ;
        std::map<std::string, unsigned int>  m_order ;
    private:
        std::vector<unsigned int>            m_color_codes ; 
        std::vector<std::string>             m_labels ; 

};

#include "OKCORE_TAIL.hh"


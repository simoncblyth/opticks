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

#include <string>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

/**
ContentStyle
===============

Canonical m_content_style instance is ctor resident of Composition

**/

class OKCORE_API ContentStyle {
   public:
        ContentStyle();
   public:
        void nextContentStyle();
        void command(const char* cmd); 

        std::string desc() const ; 
        bool isInst() const ; 
        bool isBBox() const ; 
        bool isWire() const ; 
        bool isASIS() const ; 
   public: 
        typedef enum { ASIS, BBOX, NORM, NONE, WIRE, NUM_CONTENT_STYLE, NORM_BBOX } ContentStyle_t ;
        void setNumContentStyle(unsigned num_content_style); // used to disable WIRE style for JUNO
   private:
        // ContentStyle
        static const char* ASIS_ ; 
        static const char* BBOX_ ; 
        static const char* NORM_ ; 
        static const char* NONE_ ; 
        static const char* WIRE_ ; 
        static const char* NORM_BBOX_ ; 

        unsigned getNumContentStyle(); // allows ro override the enum
        void setContentStyle(ContentStyle::ContentStyle_t style);
        ContentStyle::ContentStyle_t getContentStyle() const ; 
        void applyContentStyle();
        static const char* getContentStyleName(ContentStyle::ContentStyle_t style);
        const char* getContentStyleName() const ;
        void dumpContentStyles(const char* msg); 
   private:
        ContentStyle_t  m_content_style ; 
        unsigned int    m_num_content_style ; 
        bool            m_inst ; 
        bool            m_bbox ; 
        bool            m_wire ; 
        bool            m_asis ; 

};


#include "OKCORE_TAIL.hh"


 

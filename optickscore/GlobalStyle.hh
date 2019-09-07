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
GlobalStyle
===============

Canonical m_global_style instance is ctor resident of Composition

**/
 
class OKCORE_API GlobalStyle
{
    public:
        GlobalStyle(); 

        static const char* GVIS_ ; 
        static const char* GINVIS_ ; 
        static const char* GVISVEC_ ; 
        static const char* GVEC_ ; 
        static const char* GlobalStyleName( int style ); 

        const char* getGlobalStyleName() const ;   
        std::string desc() const ; 

        typedef enum { GVIS, GINVIS, GVISVEC, GVEC, NUM_GLOBAL_STYLE } GlobalStyle_t ;  
        unsigned int getNumGlobalStyle(); 
        void setNumGlobalStyle(unsigned int num_global_style); // used to disable GVISVEC GVEC styles for JUNO

        void nextGlobalStyle();  
        void command(const char* cmd) ;


        void setGlobalStyle(int style); 
        void applyGlobalStyle();
       

        bool* getGlobalModePtr(); 
        bool* getGlobalVecModePtr(); 

    private:
        bool         m_global_mode ; 
        bool         m_globalvec_mode ; 
        
        GlobalStyle_t   m_global_style ; 
        unsigned int    m_num_global_style ; 



};

#include "OKCORE_TAIL.hh"


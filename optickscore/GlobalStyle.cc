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

#include <cassert>
#include <sstream>

#include "PLOG.hh"
#include "GlobalStyle.hh"


const char* GlobalStyle::GVIS_ = "GVIS" ; 
const char* GlobalStyle::GINVIS_ = "GINVIS"; 
const char* GlobalStyle::GVISVEC_ = "GVISVEC" ; 
const char* GlobalStyle::GVEC_ = "GVEC" ; 
 
const char* GlobalStyle::GlobalStyleName(int style)
{
    const char* s = NULL ; 
    switch((GlobalStyle_t)style)
    {
        case GVIS  :  s = GVIS_    ; break ; 
        case GINVIS:  s = GINVIS_  ; break ; 
        case GVISVEC: s = GVISVEC_ ; break ; 
        case GVEC:    s = GVEC_    ; break ; 
        default:      s = NULL     ; break ; 
    } 
    assert(s); 
    return s ; 
}

const char* GlobalStyle::getGlobalStyleName() const 
{
    return GlobalStyleName(m_global_style) ; 
}    


std::string GlobalStyle::desc() const 
{
    std::stringstream ss ; 
    ss << "GlobalStyle "
       << getGlobalStyleName() 
       ;
    return ss.str(); 
}



GlobalStyle::GlobalStyle()
    :
    m_global_mode(false),
    m_globalvec_mode(false),
    m_global_style(GVIS),
    m_num_global_style(GVISVEC)
{
}

bool* GlobalStyle::getGlobalModePtr(){    return &m_global_mode ; }
bool* GlobalStyle::getGlobalVecModePtr(){ return &m_globalvec_mode ; }


void GlobalStyle::command(const char* cmd) 
{
    assert( strlen(cmd) == 2 ); 
    assert( cmd[0] == 'Q' ); 

    GlobalStyle_t style = GVIS ;  

    switch( cmd[1] )
    {  
        case '0': style = GVIS       ; break ; 
        case '1': style = GINVIS     ; break ; 
        case '2': style = GVISVEC    ; break ; 
        case '3': style = GVEC       ; break ; 
        default:  assert(0)          ; break ; 
    }

    setGlobalStyle(style);
}


unsigned int GlobalStyle::getNumGlobalStyle()
{
    return m_num_global_style == 0 ? int(NUM_GLOBAL_STYLE) : m_num_global_style ;
}
void GlobalStyle::setNumGlobalStyle(unsigned int num_global_style)
{
    m_num_global_style = num_global_style ;
}

void GlobalStyle::nextGlobalStyle()
{
    int next = (m_global_style + 1) % getNumGlobalStyle() ; 
    setGlobalStyle(next); 
}

void GlobalStyle::setGlobalStyle( int style ) 
{
    m_global_style = (GlobalStyle_t)style ; 
    applyGlobalStyle();
    LOG(info) << desc() ; 
}



void GlobalStyle::applyGlobalStyle()
{
   // { GVIS, 
   //   GINVIS, 
   //   GVISVEC, 
   //   GVEC, 
   //   NUM_GLOBAL_STYLE }


    switch(m_global_style)
    {
        case GVIS:
                  m_global_mode = true ;    
                  m_globalvec_mode = false ;    
                  break ; 
        case GVISVEC:
                  m_global_mode = true ;    
                  m_globalvec_mode = true ;
                  break ; 
        case GVEC:
                  m_global_mode = false ;    
                  m_globalvec_mode = true ;
                  break ; 
        case GINVIS:
                  m_global_mode = false ;    
                  m_globalvec_mode = false ;
                  break ; 
        default:
                  assert(0);
        
    }
}





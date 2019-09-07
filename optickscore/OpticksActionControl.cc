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

#include "OpticksActionControl.hh"
#include <sstream>
#include "BStr.hh"
#include "PLOG.hh"

const char* OpticksActionControl::GS_LOADED_     = "GS_LOADED" ; 
const char* OpticksActionControl::GS_FABRICATED_ = "GS_FABRICATED" ; 
const char* OpticksActionControl::GS_TRANSLATED_ = "GS_TRANSLATED" ; 
const char* OpticksActionControl::GS_TORCH_      = "GS_TORCH" ; 
const char* OpticksActionControl::GS_LEGACY_     = "GS_LEGACY" ; 
const char* OpticksActionControl::GS_EMBEDDED_   = "GS_EMBEDDED";
const char* OpticksActionControl::GS_EMITSOURCE_     = "GS_EMITSOURCE";

std::vector<const char*> OpticksActionControl::Tags()
{
    std::vector<const char*> tags ; 
    tags.push_back(GS_LOADED_);
    tags.push_back(GS_FABRICATED_);
    tags.push_back(GS_TRANSLATED_);
    tags.push_back(GS_TORCH_);
    tags.push_back(GS_LEGACY_);
    tags.push_back(GS_EMBEDDED_);
    tags.push_back(GS_EMITSOURCE_);
    return tags  ;
}

std::string OpticksActionControl::Description(unsigned long long ctrl)
{
   std::stringstream ss ;
   if( ctrl & GS_LOADED )      ss << GS_LOADED_ << " "; 
   if( ctrl & GS_FABRICATED  ) ss << GS_FABRICATED_ << " "; 
   if( ctrl & GS_TRANSLATED )  ss << GS_TRANSLATED_ << " "; 
   if( ctrl & GS_TORCH)        ss << GS_TORCH_ << " "; 
   if( ctrl & GS_LEGACY)       ss << GS_LEGACY_ << " ";
   if( ctrl & GS_EMBEDDED)     ss << GS_EMBEDDED_ << " ";
   if( ctrl & GS_EMITSOURCE)       ss << GS_EMITSOURCE_ << " ";
   return ss.str();
}

unsigned long long OpticksActionControl::ParseTag(const char* k)
{
    unsigned long long tag = 0 ;
    if(     strcmp(k,GS_LOADED_)==0)          tag = GS_LOADED ;
    else if(strcmp(k,GS_FABRICATED_)==0)      tag = GS_FABRICATED ;
    else if(strcmp(k,GS_TRANSLATED_)==0)      tag = GS_TRANSLATED ;
    else if(strcmp(k,GS_TORCH_)==0)           tag = GS_TORCH ;
    else if(strcmp(k,GS_LEGACY_)==0)          tag = GS_LEGACY ;
    else if(strcmp(k,GS_EMBEDDED_)==0)        tag = GS_EMBEDDED;
    else if(strcmp(k,GS_EMITSOURCE_)==0)          tag = GS_EMITSOURCE;
    return tag ;
}


unsigned long long OpticksActionControl::Parse(const char* ctrl_, char delim)
{
    unsigned long long ctrl(0) ; 
    if(ctrl_)
    {
        std::vector<std::string> elems ; 
        BStr::split(elems,ctrl_,delim);
        for(unsigned i=0 ; i < elems.size() ; i++)
        {
            const char* tag_ = elems[i].c_str() ;
            unsigned long long tag = ParseTag(tag_) ;
            if(tag == 0)
                 LOG(fatal) << "OpticksActionControl::Parse BAD TAG " << tag_ ;
            assert(tag);
            ctrl |= tag ;
        }
    }
    return ctrl ; 
}
bool OpticksActionControl::isSet(unsigned long long ctrl, const char* mask_)  
{
    unsigned long long mask = Parse(mask_) ;   
    bool match = (ctrl & mask) != 0 ; 
    return match ; 
}

OpticksActionControl::OpticksActionControl(unsigned long long* ctrl)
    :
    m_ctrl(ctrl)
{
}


void OpticksActionControl::add(const char* ctrl)
{
    *m_ctrl |= Parse(ctrl) ;
}

bool OpticksActionControl::isSet(const char* mask) const 
{
    return isSet(*m_ctrl, mask );
}

bool OpticksActionControl::operator()(const char* mask) const 
{
    return isSet(*m_ctrl, mask );
}



std::string OpticksActionControl::description(const char* msg) const
{
   std::stringstream ss ;
   ss << msg << " : " ;
   ss << Description(*m_ctrl) ;
   return ss.str();
}

 


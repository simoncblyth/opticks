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

#include "NQuad.hpp"

class BMeta ; 

#include <map>
#include <string>
#include <vector>

#include "plog/Severity.h"

template <typename T> class NPY ; 

/*

*OpticksColors* associates color names to RGB hexcodes

::

    delta:~ blyth$ cat ~/.opticks/GCache/GColors.json | tr "," "\n"
    {"indigo": "#4B0082"
     "gold": "#FFD700"
     "hotpink": "#FF69B4"
     "firebrick": "#B22222"
     "indianred": "#CD5C5C"
     "yellow": "#FFFF00"
     "mistyrose": "#FFE4E1"
     "darkolivegreen": "#556B2F"


The codes are obtained from matplotlib::

    import json, matplotlib.colors  
    json.dump(matplotlib.colors.cnames, open("/tmp/colors.json", "w"))

The composite color buffer is avaiable as a texture sampler on the GPU
this is put togther in GLoader::load

*/



#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksColors {  
      static const plog::Severity LEVEL ; 
   public:
        enum {
           MATERIAL_COLOR_OFFSET    = 0,  
           FLAG_COLOR_OFFSET        = 64,
           PSYCHEDELIC_COLOR_OFFSET = 96,
           SPECTRAL_COLOR_OFFSET    = 256,
           COLORMAX                 = 256
       };
   public:
       static const char* NAME ; 
       static const char* COLORMAP_NAME2HEX ; 
       static OpticksColors* load(const char* dir, const char* name=NAME);
       static OpticksColors* LoadMeta();
   public:
       OpticksColors();

       void sort();
       void dump(const char* msg="OpticksColors::dump");
       void test(const char* msg="OpticksColors::test");
       bool operator() (const std::string& a, const std::string& b);

       static nvec3 makeColor( unsigned int rgb );

       const char* getNamePsychedelic(unsigned int index);
       unsigned int getCode(const char* name, unsigned int missing=0xFFFFFF);
       const char* getHex( const char* name, const char* missing=NULL);
       const char* getName(const char* hex_, const char* missing=NULL);
       unsigned int parseHex(const char* hex_);

       nvec3 getColor(const char* name, unsigned int missing=0xFFFFFF);
       nvec3 getPsychedelic(unsigned int num);
       std::vector<unsigned int>& getPsychedelicCodes() ;
       std::vector<unsigned int>& getSpectralCodes() ;

       unsigned int getNumColors();
       unsigned int getNumBytes();
       unsigned int getBufferEntry(unsigned char* colors);
   public: 
       NPY<unsigned char>* make_buffer();
       NPY<unsigned char>* make_buffer(std::vector<unsigned int>& codes);
   public: 
       void setupCompositeColorBuffer(std::vector<unsigned int>&  material_codes, std::vector<unsigned int>& flag_codes);
       NPY<unsigned char>* getCompositeBuffer();
       nuvec4   getCompositeDomain();
       void dumpCompositeBuffer(const char* msg="OpticksColors::dumpCompositeBuffer");

   private:
       void initCompositeColorBuffer(unsigned int max_colors);
       void addColors(std::vector<unsigned int>& codes, unsigned int offset=0 );
       void loadMaps(const char* dir);
       void loadMeta(BMeta* meta);

   private:
       std::vector<std::string>            m_psychedelic_names ; 
       std::vector<unsigned int>           m_psychedelic_codes ;
       std::vector<unsigned int>           m_spectral_codes ;
       std::map<std::string, std::string>  m_name2hex ;        // colormap
       NPY<unsigned char>*                 m_composite ;
       nuvec4                              m_composite_domain ; 

};

#include "OKCORE_TAIL.hh"



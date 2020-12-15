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

#include <vector>
class BMeta ; 
class Opticks ; 
class GItemList ;
class GSource ; 
template <typename T> class GPropertyMap ;

#include "plog/Severity.h"
#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GSourceLib
============

Examples of a source is the D65 standard illuminant. 

Used via an inverse CDF GPU texture to generate wavelengths 
that adhere to the blackbody D65 distribution

oxrap/cu/wavelength_lookup.cu::

     51 static __device__ __inline__ float source_lookup(float u)
     52 {
     53     float ui = u/source_domain.z + 0.5f ;
     54     return tex2D(source_texture, ui, 0.5f );  // line 0
     55 }
     56 
     57 static __device__ __inline__ void source_check()
     58 {
     59     float nm_a = source_lookup(0.0f);
     60     float nm_b = source_lookup(0.5f);
     61     float nm_c = source_lookup(1.0f);
     62     rtPrintf("source_check nm_a %10.3f %10.3f %10.3f  \n",  nm_a, nm_b, nm_c );
     63 }

**/

class GGEO_API GSourceLib : public GPropertyLib {
    public:
        static const plog::Severity LEVEL ; 
        static const unsigned int icdf_length ; 
        static const char* radiance_ ; 
        void save();
        static GSourceLib* load(Opticks* cache);
    public:
        GSourceLib(Opticks* cache);
    public:
        void add(GSource* source);
        unsigned int getNumSources();
    public:
        void generateBlackBodySample(unsigned int n=500000);
    public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       void sort();
       NPY<float>* createBuffer();
       BMeta*      createMeta();
       GItemList*  createNames();
    private:
        void init();
    public:
        GProperty<float>* constructSourceCDF(GPropertyMap<float>* pmap);
        GProperty<float>* constructInvertedSourceCDF(GPropertyMap<float>* pmap);
    private:
        std::vector<GPropertyMap<float>*> m_source ; 

};

#include "GGEO_TAIL.hh"


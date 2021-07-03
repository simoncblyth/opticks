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
#include "plog/Severity.h"

class BMeta ; 
class Opticks ; 
class GItemList ;
class GMaterialLib ; 
class GMaterial ; 

template <typename T> class GPropertyMap ;

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GScintillatorLib : public GPropertyLib {
    public:
        static const plog::Severity LEVEL ; 
    public:
        static const char* slow_component; 
        static const char* fast_component; 
        static const char* keyspec ;
    public:
        void save();
        void dump(const char* msg="GScintillatorLib::dump");
        static GScintillatorLib* load(Opticks* cache);
    public:
        GScintillatorLib(Opticks* cache, unsigned int icdf_length=4096);
        void Summary(const char* msg="GScintillatorLib::Summary");
    public:
        void add(GPropertyMap<double>* scint);
        unsigned int getNumScintillators();
    public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<double>* defaults); 
       void import();
       void sort();
       NPY<double>* createBuffer();
       BMeta*      createMeta();
       GItemList*  createNames();
    private:
        void init();

    public:
        // current approach uses external Geant4 based processing to yield the ICDF
        // with GScintillatorLib relegated to only handling the persisting of the ICDF 
        // via the normal GPropertyLib buffer mechanism 
        void               setGeant4InterpolatedICDF( NPY<double>* g4icdf );
        NPY<double>*       getGeant4InterpolatedICDF() const ; 
        GItemList*         geant4ICDFCreateNames() const  ; 
    public:
        NPY<double>*       legacyCreateBuffer() const ;
        GProperty<double>* legacyConstructReemissionCDF(GPropertyMap<double>* pmap) const ;
        GProperty<double>* legacyConstructInvertedReemissionCDF(GPropertyMap<double>* pmap) const ;
        GItemList*         legacyCreateNames() const ;
    public:
        //void       prepare();
        //void       dumpScintillatorMaterials(const char* msg);
        //unsigned   getNumScintillatorMaterials();
        //GMaterial* getScintillatorMaterial(unsigned int index);
    private:
        const GMaterialLib*       m_mlib ; 
        unsigned                  m_icdf_length ; 
        //std::vector<GMaterial*>   m_scintillators_raw ; // TODO: eliminate GPropertyLib already has vector of raw pmaps ?
        NPY<double>*              m_g4icdf ; 

};

#include "GGEO_TAIL.hh"


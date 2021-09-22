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
#include <map>
#include <string>


#include "G4MaterialPropertyVector.hh"
class CVec ; 
class G4MaterialPropertiesTable ;
template <typename T> class GProperty ; 
template <typename T> class NPY ; 

/**
CMPT
=====




**/

#include "CFG4_API_EXPORT.hh"
class CFG4_API CMPT {
   public:
       static CMPT* MakeDummy(); 

       static std::string Digest(G4MaterialPropertiesTable* mpt); 

       static void AddDummyProperty(G4MaterialPropertiesTable* mpt, const char* lkey, unsigned nval) ;
       static void AddConstProperty(G4MaterialPropertiesTable* mpt, const char* lkey, double pval ) ;
       static bool HasProperty(const G4MaterialPropertiesTable* mpt, const char* pname); 
   public:
       void addDummyProperty(const char* lkey, unsigned nval) ; 
       //void dump(const char* msg); 
   public:
       CMPT(G4MaterialPropertiesTable* mpt, const char* name=NULL);
       std::string digest() const ;
       bool hasProperty(const char* pname) const ; 
   public:
       // adds domain and value of prop, assumes prop domain is in ascending nm 
       // (as typically used for optical properties) and thus the order is  
       // reversed to yield energy ascending order and domain is converted to eV 
       void addProperty(const char* lkey,  GProperty<double>* prop, bool spline);
   public:
       void dumpRaw(const char* lkey);
   public:
       static void Dump(G4MaterialPropertiesTable* mpt, const char* msg="CMPT::Dump"); 
       static void Dump_OLD(G4MaterialPropertiesTable* mpt, const char* msg="CMPT::Dump"); 
       void dump(const char* msg="CMPT::dump") const; 
       void dumpProperty(const char* lkey);
       void sample(NPY<double>* a, unsigned offset, const char* _keys, double low, double step, unsigned nstep );
       void sampleSurf(NPY<double>* a, unsigned offset, double low, double step, unsigned nstep, bool specular );

       GProperty<double>* makeProperty(const char* key, double low, double step, unsigned nstep);
       G4MaterialPropertyVector* getVec(const char* lkey) const ;
       CVec* getCVec(const char* lkey) const ;

       unsigned splitKeys(std::vector<std::string>& keys, const char* _keys);
       unsigned getVecLength(const char* _keys);
       NPY<double>* makeArray(const char* _keys, bool reverse=true);


       std::string description(const char* msg);
       std::vector<std::string> getPropertyKeys() const ;

       std::vector<std::string> getPropertyDesc() const ;

       std::vector<std::string> getConstPropertyKeys() const ;

       std::vector<double> getConstPropertyValues() const ;

   private:
       G4MaterialPropertiesTable* m_mpt ; 
       const char* m_name ; 
};




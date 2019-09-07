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

template <typename T> class NPY ; 
class Typ ; 
class Types ; 
class RecordsNPY ; 
class Index ; 

// detailed host based photon and record dumper 

#include "NPY_API_EXPORT.hh"
class NPY_API PhotonsNPY {
   public:  
       PhotonsNPY(NPY<float>* photons); 
   public:  
       void                  setTypes(Types* types);
       void                  setTyp(Typ* typ);
       void                  setRecs(RecordsNPY* recs);
   public:  
       NPY<float>*           make_pathinfo();
       NPY<float>*           getPhotons();
       RecordsNPY*           getRecs();
       Types*                getTypes();

   public:  
       void dump(unsigned int photon_id, const char* msg="PhotonsNPY::dump");
   public:  
       void dumpPhotonRecord(unsigned int photon_id, const char* msg="phr");
       void dumpPhoton(unsigned int i, const char* msg="pho");
   public:  
       void dumpPhotons(const char* msg="PhotonsNPY::dumpPhotons", unsigned int ndump=5);
   public:
       void debugdump(const char* msg);

   private:
       NPY<float>*                  m_photons ; 
       bool                         m_flat ;
       RecordsNPY*                  m_recs ; 
       Types*                       m_types ; 
       Typ*                         m_typ ; 
       unsigned int                 m_maxrec ; 

};


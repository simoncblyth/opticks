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

/**
OpIndexerApp
===============

For standalone indexing tests ?

**/


class Opticks ;   // okc-
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

class OpticksHub ;   // okg-
class OpticksRun ; 

class OContext ;   // optixrap-
class OScene ; 
class OEvent ; 

class OpIndexer ;   // opop-

#include "OKOP_API_EXPORT.hh"
class OKOP_API OpIndexerApp {
   public:
      OpIndexerApp(int argc, char** argv);
   public:
      void loadEvtFromFile();
      void makeIndex();
   private:
      Opticks*              m_ok ;   
      OpticksCfg<Opticks>*  m_cfg ;
      OpticksHub*           m_hub ;   
      OpticksRun*           m_run ;   

      OScene*               m_scene ; 
      OContext*             m_ocontext ; 
      OEvent*               m_oevt ; 

      OpIndexer*            m_indexer ; 

};



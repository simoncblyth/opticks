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
#include "OKOP_API_EXPORT.hh"

/**
OpEvt
======

Light weight "API" event providing genstep manipulation.

Canonical m_opevt instance is resident of OpMgr and 
is instanciated when OpMgr::addGenstep is called.


DevNote
---------

* hmm :doc:`/cfg4/CGenstepCollector` does all that this does and more (but not too much more)

**/

template <typename T> class NPY  ; 

class OKOP_API OpEvt {
    public:
         OpEvt();
         void addGenstep( float* data, unsigned num_float );
         unsigned getNumGensteps() const ; 
         NPY<float>* getEmbeddedGensteps();

         void saveEmbeddedGensteps(const char* path) const ;
         void loadEmbeddedGensteps(const char* path);

         void resetGensteps();
    private:          
         NPY<float>* m_genstep ; 

};
 

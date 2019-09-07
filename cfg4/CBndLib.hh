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

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class OpticksHub ; 
class GBndLib ; 
class GMaterialLib ; 
class GSurfaceLib ; 

class GMaterial ; 

//class GSurLib ; 
//class GSur ; 

/**
CBndLib
===============

Q: why ? 

Eliminate this class : it does too little

**/

class CFG4_API CBndLib  
{
    public:
        CBndLib(OpticksHub* hub);
        unsigned addBoundary(const char* spec);
    public:
        GMaterial*        getOuterMaterial(unsigned boundary);
        GMaterial*        getInnerMaterial(unsigned boundary);
        GPropertyMap<float>* getOuterSurface(unsigned boundary);
        GPropertyMap<float>* getInnerSurface(unsigned boundary);

        //GSur*             getOuterSurface(unsigned boundary);
        //GSur*             getInnerSurface(unsigned boundary);
    private:
        OpticksHub*      m_hub ; 
        GBndLib*         m_blib ; 
        GMaterialLib*    m_mlib ; 
        GSurfaceLib*     m_slib ; 
        //GSurLib*         m_slib ; 

};


#include "CFG4_TAIL.hh"


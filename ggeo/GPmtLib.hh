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

#include <string>

class Opticks ; 
class GBndLib ; 
class GPmt ; 
class GMergedMesh ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**

GPmtLib
==========

DIRTY ASSOCIATION BETWEEN OLD STYLE ANALYTIC GPmt AND TRIANGULATED GMergedMesh 
    
GPmt 
  detdesc parsed analytic geometry (see pmt-ecd dd.py tree.py etc..)
    

**/

class GGEO_API GPmtLib {
        friend class CTestDetector ;  // for getLoadedAnalyticPmt
    public:
        //void save();
        static const char* TRI_PMT_PATH ; 
        static GPmtLib* load(Opticks* ok, GBndLib* bndlib);
    public:
        GPmtLib(Opticks* ok, GBndLib* bndlib);
        GMergedMesh* getPmt() ;
    private:
        GPmt* getLoadedAnalyticPmt();
    private:
        void loadAnaPmt();
        void loadTriPmt();
        void dirtyAssociation();
        std::string getTriPmtPath();
    private:
        Opticks*     m_ok ; 
        GBndLib*     m_bndlib ; 
    private:
        GPmt*        m_apmt ; 
        GMergedMesh* m_tpmt ; 
 
};

#include "GGEO_TAIL.hh"


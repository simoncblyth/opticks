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
#include <map>
#include "plog/Severity.h"

struct NLODConfig ; 
class Opticks ; 

class GBndLib ; 
class GMergedMesh ; 
class GNode ; 
class GGeoLib ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GGeoLib
==========

Container for GMergedMesh and associated GParts that handles persistency
of the objects, including their association.


Instances::

    simon:opticks blyth$ opticks-find GGeoLib\(  | grep new
    ./ggeo/GGeo.cc:   m_geolib = new GGeoLib(m_ok, m_analytic, m_bndlib );
    ./ggeo/GGeoLib.cc:    GGeoLib* glib = new GGeoLib(opticks, analytic, bndlib);
    ./ggeo/GScene.cc:    m_geolib(loaded ? GGeoLib::Load(m_ok, m_analytic, m_tri_bndlib )   : new GGeoLib(m_ok, m_analytic, m_tri_bndlib)),


*/

class GGEO_API GGeoLib {
    public:
        static const plog::Severity LEVEL ;  
        static const char* GMERGEDMESH ; 
        static const char* GPARTS ; 
        static const char* GPTS ; 
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        static bool HasCacheConstituent(const char* idpath, unsigned ridx) ;
        static GGeoLib* Load(Opticks* ok, GBndLib* bndlib);
    public:
        GGeoLib(Opticks* ok, GBndLib* bndlib);

        GBndLib* getBndLib() const ; 
        std::string desc() const ; 
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion() const ;
        unsigned getVerbosity() const ;  
        int checkMergedMeshes() const ; 
    public:
        void dryrun_convert(); 
        void dryrun_convertMergedMesh(unsigned i);
        void dryrun_makeGlobalGeometryGroup(GMergedMesh* mm);
        void dryrun_makeRepeatedAssembly(GMergedMesh* mm);
        void dryrun_makeOGeometry(GMergedMesh* mm);
        void dryrun_makeGeometryTriangles(GMergedMesh* mm);
        void dryrun_makeAnalyticGeometry(GMergedMesh* mm);
        void dryrun_makeTriangulatedGeometry(GMergedMesh* mm);
    public:
        void hasCache() const ;
        void loadFromCache();
        void save();
        GMergedMesh* makeMergedMesh(unsigned index, const GNode* base, const GNode* root, unsigned verbosity, bool globalinstance );
    private:
        void loadConstituents(const char* idpath);
        void removeConstituents(const char* idpath);
        void saveConstituents(const char* idpath);
    public:
        void dump(const char* msg="GGeoLib::dump");
        unsigned getNumMergedMesh() const ;
        GMergedMesh* getMergedMesh(unsigned index) const ;
        void setMergedMesh(unsigned int index, GMergedMesh* mm);
        void eraseMergedMesh(unsigned int index);
        void clear();
    private:
        Opticks* m_ok ; 
        NLODConfig* m_lodconfig ; 
        int      m_lod ;  
        GBndLib* m_bndlib ; 
        char*   m_mesh_version ;
        int     m_verbosity ;
        std::map<unsigned,GMergedMesh*>  m_merged_mesh ; 
        GGeoLib* m_geolib ; 
};

#include "GGEO_TAIL.hh"


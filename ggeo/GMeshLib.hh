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

#include <map>
#include <string>
#include <vector>
#include <ostream>
#include "plog/Severity.h"

class Opticks ; 

class GItemList ;  
class GMesh ; 
class NCSG ; 


#include "GGEO_API_EXPORT.hh"

/*

GMeshLib : provides load/save for GMesh instances with associated names
==========================================================================

* canonical m_meshlib instances are constituents of GGeo and GScene.
* manages a vector of GMesh* and a GItemIndex of names

*/


class GGEO_API GMeshLib 
{
        friend class GGeo ; 
        friend class GScene ; 
    public:
        static const plog::Severity LEVEL ;  
        static const unsigned MAX_MESH  ; 

        static const char*    GMESHLIB ; 
        static const char*    GMESHLIB_LIST ; 
        static const char*    GMESHLIB_NCSG ; 

        static GMeshLib* Load(Opticks* ok );
    public:
        GMeshLib(Opticks* ok); 
        void add(const GMesh* mesh, bool alt=false );
        void replace(unsigned index, GMesh* replacement ); 
    public:
        std::string operator()( const char* arg ) const  ;
        void dump(const char* msg="GMeshLib::dump") const;
    public:
        // methods working from the index, so work prior to loading meshes
        const char* getMeshName(unsigned aindex) const ; 
        void        getMeshNames(std::vector<std::string>& meshNames) const ;
    public:
        std::string desc() const ; 
        unsigned    getNumMeshes() const ; 

        void          getMeshIndicesWithAlt(std::vector<unsigned>& indices) const ; 
        const GMesh*  getAltMesh(unsigned index) const ; 
        GMesh*        getMeshSimple(unsigned index) const ;  

        int           getMeshIndexWithName(const char* name, bool startswith) const ;
        const GMesh*  getMeshWithIndex(unsigned aindex) const ;  // first mesh in m_meshes addition order with getIndex() matching aindex 
        const GMesh*  getMeshWithName(const char* name, bool startswith) const ;
        const NCSG*   getSolidWithIndex(unsigned aindex) const ;  // first mesh in m_solids addition order with getIndex() matching aindex 

        const std::vector<const NCSG*>& getSolids() const ; 
        const std::vector<const GMesh*>& getMeshes() const ; 
    private:
        int         findMeshIndex( const GMesh* mesh ) const ; 
        int         findSolidIndex( const NCSG* solid ) const ; 
        void        loadFromCache();
        void        save() ; 
    private:
        void removeDirs(const char* idpath ) const ;
    private:
        void saveMeshes(const char* idpath) const ;
        void loadMeshes(const char* idpath ) ;
        void addAltMeshes(); 
        void saveAltReferences(); 
        void loadAltReferences(); 
    private:
        unsigned getNumSolids() const ;  // should give same as getNumMeshes
        void loadSolids(const char* idpath ) ;
    public:
        std::map<unsigned,unsigned>& getMeshUsage();
        std::map<unsigned,std::vector<unsigned> >& getMeshNodes();
        void countMeshUsage(unsigned meshIndex, unsigned nodeIndex);
        void reportMeshUsage(const char* msg="GMeshLib::reportMeshUsage") const ;
        void writeMeshUsage(const char* path="/tmp/GMeshLib_MeshUsageReport.txt") const ;
        void reportMeshUsage_(std::ostream& out) const ;
        void saveMeshUsage(const char* idpath) const ;
    private:
        Opticks*                      m_ok ; 
        bool                          m_direct ;  
        const char*                   m_reldir ; 
        const char*                   m_reldir_solids ; 
        GItemList*                    m_meshnames ; 
        std::vector<const GMesh*>     m_meshes ; 
        std::vector<const NCSG*>      m_solids ; 
        std::map<unsigned, unsigned>                  m_mesh_usage ; 
        std::map<unsigned, std::vector<unsigned> >    m_mesh_nodes ; 

};

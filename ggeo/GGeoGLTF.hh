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
#include "GGEO_API_EXPORT.hh"

#include "GSolidRec.hh"

class GGeo ; 
class GVolume ; 
class GMaterialLib ; 
class GBndLib ; 

namespace YOG 
{
   struct Sc ; 
   struct Mh ; 
   struct Nd ; 
   struct Maker ; 
}

/**
GGeoGLTF
=========

Writes the glTF 2.0 representation of a GGeo geometry.
glTF is a json based 3D file format that refers to other 
binary files for vertex and triangle data. 

Issues
-------

1. Suspect geocache duplication between the glTF extras and the GMeshLib persisted GMesh,
   TODO: consolidate to avoid this : not need to use an "extras" dir a "GMeshLib" 
   dir would work just fine  
   
2. Have not tried using this postcache OR with test geometry 
   
**/

class GGEO_API GGeoGLTF
{
    public:
        static const plog::Severity LEVEL ; 
    public:
        static void Save( const GGeo* ggeo, const char* path, int root ) ; 
    public:
        GGeoGLTF( const GGeo* ggeo ); 
        void save(const char* path, int root );
    private:
        void init();
        void addMaterials();
        void addMeshes();
        void addNodes();
    private:
        void addNodes_r(const GVolume* volume, YOG::Nd* parent_nd, int depth);
    public:
        void dumpSolidRec(const char* msg="GGeoGLTF::dumpSolidRec") const ;
        void writeSolidRec(const char* dir) const ;
    private:
        void solidRecTable( std::ostream& out ) const ; 
    private:
        const GGeo*            m_ggeo ;
        const GMaterialLib*    m_mlib ; 
        const GBndLib*         m_blib ; 
        YOG::Sc*               m_sc ;
        YOG::Maker*            m_maker ;
        std::vector<GSolidRec> m_solidrec ;

};



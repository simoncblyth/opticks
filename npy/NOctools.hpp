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
#include <functional>
#include <boost/unordered_map.hpp>

#include "NGLM.hpp"
#include "NQuad.hpp"
#include "NBBox.hpp"
#include "NGrid3.hpp"

struct FGLite ; 

template <typename FVec, typename IVec, int DIM> struct NGrid ; 
template <typename FVec, typename IVec, int DIM> struct NField ; 
template <typename FVec, typename IVec> struct NFieldGrid3 ; 

typedef NFieldGrid3<glm::vec3,glm::ivec3> FG3 ; 
typedef NField<glm::vec3,glm::ivec3,3>    F3 ; 
typedef NGrid<glm::vec3,glm::ivec3,3 >    G3 ; 


class BTimeKeeper ; 
class NTrianglesNPY ; 

#include "NPY_API_EXPORT.hh"



enum { 
               BUILD_BOTTOM_UP = 0x1 << 0, 
               BUILD_TOP_DOWN  = 0x1 << 1,
               USE_BOTTOM_UP   = 0x1 << 2, 
               USE_TOP_DOWN    = 0x1 << 3, 
               BUILD_BOTH      = BUILD_BOTTOM_UP | BUILD_TOP_DOWN
      };
 







template<typename T>
class NPY_API NConstructor 
{
    static const int maxlevel = 10 ; 
    static const glm::ivec3 _CHILD_MIN_OFFSETS[8] ;

    typedef std::function<float(float,float,float)> FN ; 
    typedef boost::unordered_map<unsigned, T*> UMAP ;
    UMAP cache[maxlevel] ; 

    public:
        NConstructor(FG3* fieldgrid, FGLite* fglite, const nvec4& ce, const nbbox& bb, int nominal, int coarse, int verbosity );
        T* create();
        void dump();
        void report(const char* msg="NConstructor::report");
    public:
         // grid debugging 
        void scan(const char* msg="scan", int depth=2, int limit=30 ) const ; 
        void corner_scan(const char* msg="corner_scan", int depth=2, int limit=30) const ; 

        void dump_domain(const char* msg="dump_domain") const ;

        glm::vec3 position_ce(const glm::ivec3& offset_ijk, int depth) const ;
        float      density_ce(const glm::ivec3& offset_ijk, int depth) const ;

        glm::vec3 position_bb(const glm::ivec3& natural_ijk, int depth) const ;
        float      density_bb(const glm::ivec3& natural_ijk, int depth) const ;
    private:
        T* create_coarse_nominal();
        T* create_nominal();
        void buildBottomUpFromLeaf(int leaf_loc, T* leaf );
    private:
        NMultiGrid3<glm::vec3,glm::ivec3> m_mgrid ; 

        FG3*        m_fieldgrid ; 
        F3*         m_field ; 
        FN*         m_func ; 
        FGLite*     m_fglite ; 
        nvec4       m_ce ;  
        nbbox       m_bb ; 

        G3*         m_nominal ; 
        G3*         m_coarse ; 
        int         m_verbosity ; 
        G3*         m_subtile ; 
        G3*         m_dgrid ; 

        glm::ivec3      m_nominal_min ; 
        int         m_upscale_factor ; 

        T* m_root ; 

        unsigned m_num_leaf ; 
        unsigned m_num_from_cache ; 
        unsigned m_num_into_cache ; 
        unsigned m_coarse_corners ; 
        unsigned m_nominal_corners ; 
     
};




template<typename T>
class NPY_API NManager 
{
    public:
   public:
        NManager(const unsigned ctrl, const int nominal, const int coarse, const int verbosity, const float threshold, FG3* fieldgrid, FGLite* fglite, const nbbox& bb, BTimeKeeper* timer);

        void buildOctree();
        void simplifyOctree();
        void generateMeshFromOctree();
        NTrianglesNPY* collectTriangles();
        void meshReport(const char* msg="NManager::meshReport");

        T* getRaw();
        T* getSimplified();

    private:
        unsigned m_ctrl ; 
        int      m_nominal_size ; 
        int      m_verbosity ; 
        float    m_threshold ;
        FG3*     m_fieldgrid ; 
        FGLite*  m_fglite ; 

        nbbox    m_bb ; 
        BTimeKeeper*   m_timer ;    

        nvec4        m_ce ; 
        NConstructor<T>* m_ctor ; 

        T*  m_bottom_up ;         
        T*  m_top_down ;         
        T*  m_raw ;         
        T*  m_simplified ;         
        std::vector<glm::vec3> m_vertices;
        std::vector<glm::vec3> m_normals;
        std::vector<int>       m_indices;


};



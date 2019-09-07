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

#include "NGLM.hpp"

template <typename T> class NPY ;

struct nbbox ; 
struct ntriangle ; 
struct NTriSource ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"


struct NPY_API NVtxIdx
{ 
    NPY<float>*    vtx ; 
    NPY<unsigned>* idx ; 
};

class NPY_API NTrianglesNPY {
    public:
        static const char* PLACEHOLDER ; 
    public:
        static const glm::vec3 PX ; 
        static const glm::vec3 PY ; 
        static const glm::vec3 PZ ; 
        static const glm::vec3 MX ; 
        static const glm::vec3 MY ; 
        static const glm::vec3 MZ ; 
    public:
        static const glm::vec3 Ip0 ; 
        static const glm::vec3 Ip1 ; 
        static const glm::vec3 Ip2 ; 
        static const glm::vec3 Ip3 ; 
        static const glm::vec3 Ip4 ; 
        static const glm::vec3 Ip5 ;

        static const glm::vec3 Im0 ; 
        static const glm::vec3 Im1 ; 
        static const glm::vec3 Im2 ; 
        static const glm::vec3 Im3 ; 
        static const glm::vec3 Im4 ; 
        static const glm::vec3 Im5 ;
    public:
        static const glm::vec3 PXPYPZ ; 
        static const glm::vec3 PXPYMZ ; 
        static const glm::vec3 PXMYPZ ; 
        static const glm::vec3 PXMYMZ ; 
        static const glm::vec3 MXPYPZ ; 
        static const glm::vec3 MXPYMZ ; 
        static const glm::vec3 MXMYPZ ; 
        static const glm::vec3 MXMYMZ ; 
    public:
        static NTrianglesNPY* icosahedron();
        static NTrianglesNPY* hemi_octahedron();
        static NTrianglesNPY* octahedron();
        static NTrianglesNPY* cube();
        static NTrianglesNPY* box(const nbbox& bb);
        static NTrianglesNPY* sphere(unsigned int n_polar=24, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* sphere(glm::vec4& param, unsigned int n_polar=24, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* sphere(float ctmin, float ctmax, unsigned int n_polar=24, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* from_indexed( NPY<float>* vtx, NPY<unsigned>* idx ); 
        void   to_vtxidx(NVtxIdx& vtxidx) ;  
        float maxdiff( const NTrianglesNPY* other, bool dump=false );
    public:
        static NTrianglesNPY* disk(glm::vec4& param, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* prism(const glm::vec4& param); 
    public:
        NTrianglesNPY();
        NTrianglesNPY(NPY<float>* tris, NPY<float>* normals=NULL);
        NTrianglesNPY(const NTriSource* tris) ;
    public:
        NTrianglesNPY* transform(glm::mat4& m);
        NTrianglesNPY* subdivide(unsigned int nsubdiv);
    public:
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d);
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
        void add(const ntriangle& t );
        void add(NTrianglesNPY* other);
        void addNormal(const glm::vec3& n ); // in triplicate
        void addNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c );
    public:
        NPY<float>* getTris() const ;
        NPY<float>* getNormals() const ;
        unsigned int getNumTriangles();
        nbbox* findBBox();
        void setTransform(const glm::mat4& transform);
        void setTransform(const glm::vec3& scale, const glm::vec3& translate);
        glm::mat4 getTransform();
    public:
        void setPoly(const std::string& poly);
        const std::string& getPoly();
    public:
        void setMessage(const std::string& msg);
        const std::string& getMessage();
        bool  hasMessage(const std::string& msg);
        bool  isPlaceholder();
        void dump(const char* msg="NTrianglesNPY::dump") const ;
    private:
        NPY<float>*  m_tris ; 
        NPY<float>*  m_normals ; 
        glm::mat4    m_transform ; 
        std::string  m_message ; 
        std::string  m_poly ; 
};

#include "NPY_TAIL.hh"



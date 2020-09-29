#pragma once

#include "plog/Severity.h"
#include "NGLM.hpp"
template <typename T> class NPY ;

#include "NPY_API_EXPORT.hh"

/**
NPoint
======

This is aiming to replace gfloat3/GBuffer vertices and normals in GMesh, 
but it aint simple.

3/4 difference is part of the problem.


**/

class NPY_API NPoint {
        static const plog::Severity LEVEL ; 
    public:
        static bool HasSameDigest(const NPoint* a , const NPoint* b);
        static NPoint* MakeTransformed( const NPoint* src, const glm::mat4& transform );
    public:
        NPoint* spawnTransformed( const glm::mat4& transform );
    public:
        NPoint(unsigned n); 
    private:
        void init();
    public:
        unsigned getNum() const ;
        std::string digest() const ; 
        std::string desc(unsigned i) const ;
        void dump(const char* msg="NPoint::dump") const ; 
    public:
        void add(const glm::vec3& v , float w);
        void add(const glm::vec4& q );
    public:
        void set(unsigned i, const glm::vec3& v, float w) ; 
        void set(unsigned i, float x, float y, float z, float w) ; 
        void set(unsigned i, const glm::vec4& q) const ; 
    public:
        glm::vec4 get(unsigned i) const ;
    private:
        NPY<float>*  m_arr ; 
};



 

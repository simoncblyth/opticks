#pragma once

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "NGLM.hpp"
#include "NPYBase.hpp"

/**
NPX
=====

This is a stripped down version of NPY with only essentials, useful for debugging.

**/

template <class T>
struct NPY_API NPX : public NPYBase {

    static const plog::Severity LEVEL ; 
    static NPX<T>* make(const std::vector<int>& shape);
    static NPX<T>* make(unsigned int ni, unsigned int nj, unsigned int nk );

    NPX(const std::vector<int>& shape, const T*  data, std::string& metadata) ;
       
    T* allocate();

    void setData(const T* data);
    void reset();
    void deallocate();

    void add(const T* values, unsigned int nvals);   // add values, nvals must be integral multiple of the itemsize  

    T* grow(unsigned int nitems);
    void reserve(unsigned nitems);

    void zero();
    void read(const void* ptr);

    void save(const char* path) const ;
    void save(const char* dir, const char* name) const ;
    void save(const char* pfx, const char* tfmt, const char* targ, const char* tag, const char* det ) const ;

    void* getBytes() const ;
    void setQuad(const glm::vec4& vec, unsigned int i, unsigned int j, unsigned int k) ;
    void setQuad(const glm::ivec4& vec, unsigned int i, unsigned int j, unsigned int k) ;

    glm::vec4 getQuadF( int i,  int j,  int k ) const ;
    glm::ivec4  getQuadI( int i,  int j,  int k ) const ;


    std::vector<T>* m_data ; 

};



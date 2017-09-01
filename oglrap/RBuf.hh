#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "OGLRAP_API_EXPORT.hh"

struct OGLRAP_API RBuf
{
    unsigned id ; 
    unsigned num_items ;
    unsigned num_bytes ;
    int      query_count ; 
    void*    ptr ;

    unsigned item_bytes() const ;

    RBuf(unsigned num_items_, unsigned num_bytes_, void* ptr_) ;
    RBuf* cloneNull() const ;
    RBuf* cloneZero() const ;
    
    void upload(GLenum target, GLenum usage );
    void uploadNull(GLenum target, GLenum usage );

    std::string desc() const ;
    std::string brief() const ;
    void dump(const char* msg) const ; 

    static RBuf* Make(const std::vector<glm::mat4>& mat) ;
    static RBuf* Make(const std::vector<glm::vec4>& vert) ;
    static RBuf* Make(const std::vector<unsigned>&  elem) ;

};




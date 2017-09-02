#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "OGLRAP_API_EXPORT.hh"


#define MAKE_RBUF(buf) ((buf) ? new RBuf((buf)->getNumItems(), (buf)->getNumBytes(), (buf)->getNumElements(), (buf)->getPointer() ) : NULL )


struct OGLRAP_API RBuf
{
    static const unsigned UNSET ; 

    unsigned id ; 
    unsigned num_items ;
    unsigned num_bytes ;
    unsigned num_elements ;
    int      query_count ; 
    void*    ptr ;
    bool     gpu_resident ; 

    unsigned item_bytes() const ;



    void* getPointer() const { return ptr ; } ;
    unsigned getBufferId() const { return id ; } ;
    unsigned getNumItems() const { return num_items ; } ;
    unsigned getNumBytes() const { return num_bytes ; } ;
    unsigned getNumElements() const { return num_elements ; } ;


    RBuf(unsigned num_items_, unsigned num_bytes_, unsigned num_elements_, void* ptr_) ;

    RBuf* cloneNull() const ;
    RBuf* cloneZero() const ;
    RBuf* clone() const ;
    
    void upload(GLenum target, GLenum usage );
    void uploadNull(GLenum target, GLenum usage );

    std::string desc() const ;
    std::string brief() const ;
    void dump(const char* msg) const ; 

    static RBuf* Make(const std::vector<glm::mat4>& mat) ;
    static RBuf* Make(const std::vector<glm::vec4>& vert) ;
    static RBuf* Make(const std::vector<unsigned>&  elem) ;

};




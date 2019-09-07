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
#include <vector>

#include "NGLM.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "OGLRAP_API_EXPORT.hh"

#define MAKE_RBUF(buf) ((buf) ? new RBuf((buf)->getNumItems(), (buf)->getNumBytes(), (buf)->getNumElements(), (buf)->getPointer(), (buf)->getName() ) : NULL )


struct OGLRAP_API RBuf
{
    static char* Owner ; 
    static const unsigned UNSET ; 

    unsigned id ; 

    unsigned num_items ;
    unsigned num_bytes ;
    unsigned num_elements ;
    int      query_count ; 
    void*       ptr ;
    const char* name ; 

    bool     gpu_resident ; 
    unsigned max_dump ; 
    int      debug_index ; 

    unsigned item_bytes() const ;
    bool isUploaded() const  ;

    void* getPointer() const { return ptr ; } ;
    unsigned getBufferId() const { return id ; } ;
    unsigned getNumItems() const { return num_items ; } ;
    unsigned getNumBytes() const { return num_bytes ; } ;
    unsigned getNumElements() const { return num_elements ; } ;

    RBuf(unsigned num_items_, unsigned num_bytes_, unsigned num_elements_, void* ptr_, const char* name_=NULL) ;

    RBuf* cloneNull() const ;
    RBuf* cloneZero() const ;
    RBuf* clone() const ;
    
    void upload(GLenum target, GLenum usage );
    void uploadNull(GLenum target, GLenum usage );
    void pullback(unsigned stream );
    void bind(unsigned stream );

    std::string desc() const ;
    std::string brief() const ;
    void dump(const char* msg) const ; 

    static RBuf* Make(const std::vector<glm::mat4>& mat) ;
    static RBuf* Make(const std::vector<glm::vec4>& vert) ;
    static RBuf* Make(const std::vector<unsigned>&  elem) ;

};




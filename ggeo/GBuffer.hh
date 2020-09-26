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

/*
GBuffer
=========

* ARE TRANSITIONING FROM GBuffer TO NPY<T> WHERE POSSIBLE : 
  **DO NOT USE GBuffer IN NEW DEVELOPMENTS**

* GBuffer saves using the lower level numpy.hpp rather than NPY, 
  so in principal it can be replaced by NPY : but thats delicate work
  as GBuffer is used in many places ... probably safer to 
  move to NPY adoption in GBuffer rather than replace it


eg 10 float3 vertices, where the item is regarded at the float3 

   NumBytes          10*3*4 = 120 bytes
   ItemSize             3*4 = 12 bytes
   NumElements            3      3 float elements make up the float3
   NumItems              10  =  NumBytes/ItemSize  = 120 bytes/ 12 bytes 
   NumElementsTotal      30  =  NumItems*NumElements = 10*3 
*/ 

#include <string>
struct BBufSpec ; 
struct NSlice ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GBuffer {
    public:
        GBuffer(unsigned int nbytes, void* pointer, unsigned int itemsize, unsigned int nelem, const char* name);
    public:
        void reshape(unsigned int nelem);
        // NB reshape just changes interpretation, there is no change to NumBytes or NumElementsTotal
        //    only NumItems and NumElements are changed (reversibly)
    public:
        unsigned int getNumBytes() const ;
        void*        getPointer() const ;
        const char*  getName() const ;
        BBufSpec*    getBufSpec()  ;

        unsigned int getItemSize() const ;
        unsigned int getNumElements() const ;
        unsigned int getNumItems() const ;
        unsigned int getNumElementsTotal() const ;
    public:
        void setName(const char* name); 
    public:
        bool isEqual(GBuffer* other);
        float fractionDifferent(GBuffer* other);
        void Summary(const char* msg="GBuffer::Summary") const ;
        std::string desc() const ;
    public:
        GBuffer* make_slice(const char* slice); 
        GBuffer* make_slice(NSlice* slice); 
    public:
        template<typename T>
        void dump(const char* msg="GBuffer::dump", unsigned int limit=10);

        template<typename T>
        void save(const char* path);

        template<typename T>
        void save(const char* dir, const char* name);


        template<typename T>
        static GBuffer* load(const char* path);

        template<typename T>
        static GBuffer* load(const char* dir, const char* name);
    public:
        // OpenGL related : but not requiring any headers
        void         setBufferId(int buffer_id);
        int          getBufferId() const ;  // either -1 if not uploaded, or the OpenGL buffer Id
        void         setBufferTarget(int buffer_target);
        int          getBufferTarget() const ;
        void         didUpload(); 
    protected:
         unsigned int m_nbytes ;
         void*        m_pointer ; 
         unsigned int m_itemsize ;
         unsigned int m_nelem ;
         const char*  m_name ; 
    private:
         int          m_buffer_id ; 
         int          m_buffer_target ; 
         BBufSpec*    m_bufspec ; 

}; 




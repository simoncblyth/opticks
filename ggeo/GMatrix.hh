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
#include "GBuffer.hh"

// TODO: replace with glm ?


#include "GGEO_API_EXPORT.hh"

template<typename T>
class GGEO_API GMatrix : public GBuffer 
{
   public:
       GMatrix(T _s); 
       // homogenous scaling  matrix
       GMatrix(T _x, T _y, T _z, T _s=1.0f); 
       // homogenous translate then scale matrix (ie translation not scaled)

       GMatrix();
       GMatrix(const GMatrix& m);
       GMatrix(const T* buf);
       GMatrix(
          T _a1, T _a2, T _a3, T _a4,
          T _b1, T _b2, T _b3, T _b4,
          T _c1, T _c2, T _c3, T _c4,
          T _d1, T _d2, T _d3, T _d4); 
 

       virtual ~GMatrix();

       void Summary(const char* msg="GMatrix::Summary");

       GMatrix& operator *= (const GMatrix& m); 
       GMatrix  operator *  (const GMatrix& m) const;

       T largestDiff(const GMatrix& m);

       void copyTo(T* buf);
       void* getPointer();  // override GBuffer

       bool isIdentity();



       std::string digest();
       std::string brief(unsigned int w=11);

   public:
       T a1, a2, a3, a4 ; 
       T b1, b2, b3, b4 ; 
       T c1, c2, c3, c4 ; 
       T d1, d2, d3, d4 ; 

};



typedef GMatrix<float>  GMatrixF ;



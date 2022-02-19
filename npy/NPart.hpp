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

/**
npart
=======


**/


#include "OpticksCSG.h"
#include "NPart.h"
#include "NQuad.hpp"

struct npart ; 
struct nbbox ; 

struct NPY_API npart 
{
    nquad q0 ;  // x,y,z,w (float): param 
    nquad q1 ;  // x,y,z,w (uint) -/index/boundary/flags
    nquad q2 ;  // x,y,z (float):bbmin   w(uint):typecode  
    nquad q3 ;  // x,y,z (float):bbmax   w(uint):gpack

    nquad qx ;  // <- CPU only 
 
    static unsigned VERSION ;  // 0:with bbox, 1:without bbox and with GTransforms (*1* in use, some prim like convexpolyhedron have bbox)

    void zero();
    void dump(const char* msg) const ;
    void check_bb_is_zero(OpticksCSG_t typecode) const ;
    void setTypeCode(OpticksCSG_t typecode);
    void setGTransform(unsigned gtransform_idx, bool complement=false);
    void setITransform(unsigned itransform_idx, bool complement=false);  // used by NCSG::export_srcnode
    void setBBox(const nbbox& bb);

    void setSubNum(unsigned sub_num) ; 
    void setSubOffset(unsigned sub_offset) ; 
    unsigned getSubNum() const ; 
    unsigned getSubOffset() const ; 

    void setParam(const nquad& param);
    void setParam1(const nquad& param1);
    void setParam2(const nquad& param2);
    void setParam3(const nquad& param3);

    void setParam(float x, float y, float z, float w);
    void setParam1(float x, float y, float z, float w);


    OpticksCSG_t getTypeCode();
    bool isPrimitive();   // from TypeCode



    //NB same memory used for different purposes for CSG operator nodes
    //   distinguish usage typecode or isPrimitive

    // left/right are persisted in qx (which is only available CPU side) 
    // on GPU left/right is inferred from complete binary tree buffer index

    void setLeft(unsigned left);
    void setRight(unsigned right);
    unsigned getLeft();
    unsigned getRight();

    static void traverse( npart* tree, unsigned numNodes, unsigned node );

};




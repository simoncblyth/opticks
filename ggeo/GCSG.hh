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

template <typename T> class NPY ;
class GItemList ; 
class GMergedMesh ; 

/**
GCSG
======

* **GCSG predates full CSG tree implemented in NCSG**

Provides convenient access to csg NPY buffer created with python.

Users
------

cfg4/CMaker.cc
    pieces together the Geant4 boolean solids using the info carried by GCSG

cfg4/CPropLib.cc

cfg4/CTestDetector.cc
    builds Geant4 test geometry using info carried by GCSG 

ggeo/GPmt.cc
    GCSG is a member of GPmt that is loaded from the python prepared GPmt_csg.npy  

    See also:
 
    * notes at the tail of ggeo/GPmt.cc
    * pmt-vi pmt-ecd
    * opticks/ana/pmt/csg.py 
    * opticks/ana/pmt/tree.py 


**/

#include "GGEO_API_EXPORT.hh"
class GGEO_API GCSG {
    public:
        // shapes with analytic intersection implementations 
        static const char* SPHERE_ ;
        static const char* TUBS_ ; 
        static const char* UNION_ ; 
        static const char* INTERSECTION_ ; 
        static const char* TypeName(unsigned int typecode);

    public:
        // buffer layout, must match locations in pmt-/csg.py
        // TODO: adopt OpticksCSG enum 
        enum { 
              NJ = 4,
              NK = 4,
              UNION = 10,  
              INTERSECTION = 20,
              SPHERE = 3,
              TUBS = 4
            } ;

    public:
        GCSG(NPY<float>* buffer, GItemList* materials, GItemList* lvnames, GItemList* pvnames);

        NPY<float>* getCSGBuffer();
        void dump(const char* msg="GCSG::dump");
    public:
        const char* getMaterialName(unsigned int nodeindex);
        const char* getLVName(unsigned int nodeindex);
        const char* getPVName(unsigned int nodeindex);
    public:
        unsigned int getNumItems();
    public:
        float getX(unsigned int i);
        float getY(unsigned int i);
        float getZ(unsigned int i);
        float getOuterRadius(unsigned int i);
        float getInnerRadius(unsigned int i);
        float getSizeZ(unsigned int i);
        float getStartTheta(unsigned int i);
        float getDeltaTheta(unsigned int i);
    public:
        unsigned int getTypeCode(unsigned int i);
        bool isUnion(unsigned int i);
        bool isIntersection(unsigned int i);
        bool isSphere(unsigned int i);
        bool isTubs(unsigned int i);

        unsigned int getNodeIndex(unsigned int i);  // 1-based index, 0:unset
        unsigned int getParentIndex(unsigned int i);  // 1-based index, 0:unset
        unsigned int getSpare(unsigned int i); 

        const char* getTypeName(unsigned int i);
    public:
        unsigned int getIndex(unsigned int i);
        unsigned int getNumChildren(unsigned int i);
        unsigned int getFirstChildIndex(unsigned int i);
        unsigned int getLastChildIndex(unsigned int i);
    public:
        //GMergedMesh* makeMergedMesh();
    private:
        float        getFloat(unsigned int i, unsigned int j, unsigned int k);
        unsigned int getUInt(unsigned int i, unsigned int j, unsigned int k);

    private:
        NPY<float>*        m_csg_buffer ; 
        GItemList*         m_materials ; 
        GItemList*         m_lvnames ; 
        GItemList*         m_pvnames ; 
};




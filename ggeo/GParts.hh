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

#include <map>
#include <string>
#include <vector>

#include <glm/fwd.hpp>
#include "plog/Severity.h"
#include "OpticksCSG.h"
#include "NBBox.hpp"


#define GPARTS_DEBUG 1



struct npart ; 
struct NSlice ; 
template <typename T> class NPY ;
class NPYBase ; 
template <typename T> class GMatrix ;

class NCSG ; 

struct nivec4 ; 
struct gbbox ; 
struct gfloat3 ; 

class GPts ; 
class GItemList ; 
class GBndLib ; 

/**
GParts
======= 

* handles the concatenation of analytic geometry, by combination
  of GParts instances

* creates *primitive* buffer from the *parts* buffer

* holds boundary specifications as lists of strings
  that are only converted into actual boundaries with indices pointing 
  at materials and surface by GParts::registerBoundaries which 
  is invoked by GParts::close which happens late 
  (typically within oxrap just before upload to GPU). 

  This approach was adopted to allow dynamic addition of geometry and
  boundaries, which is convenient for testing.

* the GParts name derives from the history of being used to hold single primitive
  parts of Daya Bay PMT which were created by detdesc partitioning 
  from python (pmt-/tree.py) 


Lifecycle Summary
-----------------------

* single tree(ie single solid) GParts created from NCSG by X4PhysicalVolume::convertNode 
  on first encountering an lvIdx, where they get attached to a GVolume

* single tree GParts are merged together into combination GParts by GMergedMesh::mergeVolumeAnalytic
  where placement transforms are applied with GParts::applyPlacementTransform  

* combinend GParts are uses by OGeo::makeAnalyticGeometry to GParts::close and upload the GParts
  buffers into the OptiX context on GPU 


Details on where GParts is used
-------------------------------------

Based on *opticks-fl GParts.hh*

::

    ./ggeo/CMakeLists.txt
    ./ggeo/GParts.cc
         setup 

    ./ggeo/GPmt.cc
    ./ggeo/tests/GPmtTest.cc
    ./ggeo/GScene.cc
         near dead code : to be removed     

    ./extg4/tests/X4PhysicalVolume2Test.cc
    ./extg4/tests/X4SolidTest.cc
    ./ggeo/tests/GPartsTest.cc
         tests 

    ./extg4/X4PhysicalVolume.cc

         X4PhysicalVolume::convertNode 
             within the visit of X4PhysicalVolume::convertStructure_r
             GParts::Make creates GParts instance from the NCSG 
             associated to the GMesh for the lvIdx solid.  The GParts
             are associated with the GVolume nodes of the tree.

    ./ggeo/GMergedMesh.cc
         NB for full(not test?) geometry GMergedMesh is orchestrated
         from GGeo::prepare by the GInstancer, but most action is in GMergedMesh  

         GMergedMesh::mergeMergedMesh
             GParts::add the pts from an "other" GMergedMesh into m_parts

         GMergedMesh::mergeVolume
             invokes GMergedMesh::mergeVolumeAnalytic with the pts 
             and transform associated to the GVolume.  
             The transform is the base or root relative flobal transform

         GMergedMesh::mergeVolumeAnalytic
             GParts::add the pts argument after GParts::applyPlacementTransform
             is applied to them, using transform argument

    ./ggeo/GGeoLib.cc

          GGeoLib::loadConstituents
              GParts::Load and associates them with corresponding GMergedMesh also loaded  

          GGeoLib::saveConstituents
              GParts::save  


    ./ggeo/GGeoTest.cc

         GGeoTest::importCSG
              GParts::setIndex to the collected m_meshes index           

    ./ggeo/GMaker.cc

         GMaker::makeFromMesh
              GParts::Make from NCSG instance and spec : used 
              from GGeoTest  

         GMaker::make 
         GMaker::makePrism
              GParts::Make from parameters, type and spec : used for 
              creation of simple geometries   


    ./optixrap/OGeo.cc

          OGeo::makeAnalyticGeometry
              GParts::close constructs the primBuffer, also uploads the
              various buffers prim/tran/part/identity into OptiX context on GPU 
              

Mesh-type or node-type
--------------------------

Boundary spec is a node-type-qty, not a mesh-type-qty 
so it does not belong inside GParts (a mesh-type-qty)
... there are relatively few mesh-type-qty for 
each distinct shape (~249 DYB), but much more node-type-qty (~12k DYB)
 
BUT: GParts combination means that it kinda transitions 
between mesh-type when just for a single unplaced shape
into node-type once applyPlacementTransform is used by GMergedMesh::mergeVolumeAnalytic


persisted structure detailed in GParts.rst
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* for examination of structure of multi-complete tree buffers see GParts.rst


**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GParts { 
       // users of setPrimFlag
       friend class GGeoTest ;
       friend class GPmt ; 
       friend struct CSGOptiXGGeo ;  
    public:

#ifdef GPARTS_DEBUG
       static std::vector<unsigned>* IDXS ;
       void initDebugDupeIdx();

       std::vector<unsigned> m_nix ;
       std::vector<unsigned> m_aix ;
#endif

       static int DEBUG ; 
       static void SetDEBUG(int dbg); 

       static const plog::Severity LEVEL ; 
       // conventional names for interfacing
       static const char* CONTAINING_MATERIAL ; 
       static const char* SENSOR_SURFACE ; 
       static const int NTRAN ; 


       static void BufferTags(std::vector<std::string>& tags)  ;
       static const char* BufferName(const char* tag) ;

       template<typename T>
       static NPY<T>* LoadBuffer(const char* dir, const char* tag);

    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4,
              SK = 4  
            } ;
    public:
      //
        static int     Compare(const GParts* a, const GParts* b, bool dump ); 
        static GParts* Create(const Opticks* ok, const GPts* pts, const std::vector<const NCSG*>& solids, unsigned* num_mismatch_pt=nullptr, std::vector<glm::mat4>* mismatch_placements=nullptr );

        static GParts* Make(const npart& pt, const char* spec);
        static GParts* Make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec);
        static GParts* Make(const NCSG* tree, const char* spec, unsigned ndIdx );
    public:
        static GParts* Combine(std::vector<GParts*> subs );
        static GParts* Combine(GParts* onesub );   // for consistent handling between 1 and many 
    public:
        const std::vector<GParts*>& getSubs() const ;
        unsigned getNumSubs() const ; 
        GParts*  getSub(unsigned i) const ; 

    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<unsigned>* idxBuf, NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<unsigned>* idxBuf, NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, GItemList* spec, GBndLib* bndlib=NULL);
   public:
        void add(GParts* other );
        void close();
   public:
        void     setBndLib(GBndLib* blib);
        void     setVerbosity(unsigned verbosity); 
        unsigned getVerbosity() const ; 
        void     enlargeBBox(unsigned int part, float epsilon=0.00001f);
        void     enlargeBBoxAll(float epsilon=0.00001f);
    public:
        // transients for debugging convenience when made from NCSG
        void setCSG(const NCSG* csg);
        const NCSG* getCSG() const ;
    private:
        void init(const char* spec);        
        void init();        
        void checkSpec(GItemList* spec) const ;
    public: 
        void         setName(const char* name);
        const char*  getName() const ;

        bool isClosed() const ;
        bool isLoaded() const ;
        std::string id() const ; 

        unsigned getIndex(unsigned partIdx) const ;
        unsigned getTypeCode(unsigned partIdx) const ;
        unsigned getNodeIndex(unsigned partIdx) const ;

        unsigned getGTransform(unsigned partIdx) const ; 
        bool     getComplement(unsigned partIdx) const ; 

        unsigned getBoundary(unsigned partIdx) const ;




        unsigned  getNumTran() const ; 
        glm::mat4 getTran(unsigned tranIdx, unsigned j) const ; 

        unsigned getAnalyticVersion();
        void     setAnalyticVersion(unsigned vers);
    private: 
        void setLoaded(bool loaded=true);
    public: 
        std::string  getBoundaryName(unsigned partIdx) const ;
        std::string  getTag(unsigned partIdx) const ;
        const char*  getTypeName(unsigned partIdx) const ;
   private:
        nbbox        getBBox(unsigned i);
        gfloat3      getGfloat3(unsigned i, unsigned j, unsigned k);
        float*       getValues(unsigned i, unsigned j, unsigned k);
    public:
        nivec4       getPrimInfo(unsigned iprim) const ;
        int          getPartOffset(unsigned primIdx) const ;
        int          getNumParts(unsigned primIdx) const ;
        int          getTranOffset(unsigned primIdx) const ; 
        int          getPlanOffset(unsigned primIdx) const ;
   public:
        // late addition to assist with debugging CSGOptiXGGeo:Converter
        void setRepeatIndex(unsigned ridx); 
        unsigned getRepeatIndex() const ; 
   public:

        void setIndex(unsigned part, unsigned index);
        void setTypeCode(unsigned part, unsigned typecode);
        void setNodeIndex(unsigned part, unsigned nodeindex);  // caution slot is used for GTRANFORM index GPU side
        void setBoundary(unsigned part, unsigned boundary);
   public:
        void setBoundaryAll(unsigned boundary);
        void setNodeIndexAll(unsigned nodeindex);
    public:
        GBndLib*       getBndLib() const ;
        GItemList*     getBndSpec();
        unsigned       getNumPrim() const ;
        unsigned       getNumParts() const ;
        unsigned       getNumIdx() const ;
        unsigned       getPrimNumParts(unsigned int prim_index);
        std::string    desc(); 
    public:
        NPY<unsigned>* getIdxBuffer() const ;
        NPY<int>*      getPrimBuffer() const ;
        NPY<float>*    getPartBuffer() const ;
        NPY<float>*    getTranBuffer() const ; // inverse transforms IR*IT ie inverse of T*R 
        NPY<float>*    getPlanBuffer() const ; // planes used by convex polyhedra such as trapezoid
    public:
        NPY<float>*    getBuffer(const char* tag) const ;
        NPYBase*       getBufferBase(const char* tag) const ; 
    public:
        void fulldump(const char* msg="GParts::fulldump", unsigned lim=10 );
        void dumpPrimInfo(const char* msg="GParts::dumpPrimInfo", unsigned lim=10 );
        void dumpPrimBuffer(const char* msg="GParts::dumpPrimBuffer");
        void Summary(const char* msg="GParts::Summary", unsigned lim=10 );

        void dump(const char* msg="GParts::dump", unsigned lim=10 );
        void dumpPrim(unsigned primIdx);
        void dumpPart(unsigned partIdx);
        void dumpTran(const char* msg="GParts::dumpTran") const ; 

    public:
        void setSensorSurface(const char* surface="lvPmtHemiCathodeSensorSurface");
        void setContainingMaterial(const char* material="MineralOil");
        void applyPlacementTransform(GMatrix<float>* placement, unsigned verbosity, unsigned& num_mismatch );
        void applyPlacementTransform(const glm::mat4& placement, unsigned verbosity, unsigned& num_mismatch );

        void save(const char* dir);
        void save(const char* dir, const char* rela);
        static GParts* Load(const char* dir);
    private:
        void registerBoundaries();  // convert the boundary spec names into integer codes using bndlib, setting into partBuffer
        void makePrimBuffer();
        void reconstructPartsPerPrim();
    private:
        void setBndSpec(GItemList* bndspec);
        void setPartBuffer(NPY<float>* part_buffer);
        void setPrimBuffer(NPY<int>*   prim_buffer);
        void setIdxBuffer(NPY<unsigned>*  idx_buffer);
        void setTranBuffer(NPY<float>* tran_buffer);
        void setPlanBuffer(NPY<float>* plan_buffer);

    private:
        void         setPrimFlag(OpticksCSG_t primflag);
        OpticksCSG_t getPrimFlag() const ; 
        const char*  getPrimFlagString() const ; 
    public:
        bool isPartList() const ;
        bool isNodeTree() const ;
        bool isInvisible() const ;

        void setInvisible();
        void setPartList();
        void setNodeTree();

    public:
        const float* getPartValues(unsigned i, unsigned j, unsigned k) const ;
    private:
       unsigned int getUInt(unsigned part, unsigned j, unsigned k) const ;
       void         setUInt(unsigned part, unsigned j, unsigned k, unsigned value);

    public:
        // idx_buffer 
        //     for global pieces of geometry its useful to keep
        //      reference to the volume index at analytic level 
        static const unsigned VOL_IDX ; 
        void setVolumeIndex(unsigned idx); 
        unsigned getVolumeIndex(unsigned i) const ; 
    private:
        void     setUIntIdx( unsigned i, unsigned j, unsigned idx) ; 
        unsigned getUIntIdx( unsigned i, unsigned j ) const ; 

    private:
        // almost no state other than buffers, just icing on top of them
        // allowing this to copied/used on GPU in cu/hemi-pmt.cu
        NPY<unsigned>*     m_idx_buffer ; 
        NPY<float>*        m_part_buffer ; 
        NPY<float>*        m_tran_buffer ; 
        NPY<float>*        m_plan_buffer ; 
        GItemList*         m_bndspec ;  
        GBndLib*           m_bndlib ;   // cannot be const as registerBoundaries may add
        const char*        m_name ;         
    private:
        typedef std::map<unsigned, unsigned> MUU ; 
        typedef std::vector<unsigned> VU ; 
    private:
        NPY<int>*          m_prim_buffer ; 
        bool               m_closed ; 
        bool               m_loaded ; 
        MUU                m_parts_per_prim ;
        VU                 m_tran_per_add ; 
        VU                 m_part_per_add ; 
        VU                 m_plan_per_add ; 
        unsigned           m_verbosity ; 
        unsigned           m_analytic_version ; 
        OpticksCSG_t       m_primflag ; 
        const char*        m_medium ; 
        const NCSG*        m_csg ; 

        std::vector<GParts*> m_subs ; 
        unsigned           m_ridx ; 


};

#include "GGEO_TAIL.hh"



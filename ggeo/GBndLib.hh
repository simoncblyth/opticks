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
#include <vector>
#include <string>
#include "plog/Severity.h"

class NMeta ; 
class Opticks ; 

struct guint4 ; 
template <typename T> class GPropertyMap ;
class GMaterialLib ; 
class GSurfaceLib ; 

/**
GBndLib
=========

*GBndLib* differs from *GMaterialLib* and *GSurfaceLib* in that 
creation of its float buffer is deferred post cache 
to allow dynamic addition of boundaries for eg analytic
geometry inside-outs and for test boxes 

Buffers
---------

index_buffer
    guint4 material and surface indices imat/omat/isur/osur

optical_buffer
    optical surface property integers

boundary_buffer
     float4 memcopy zip of material and surface property buffers, 
     used to construct GPU texture.  
     Created dynamically by pulling the relevant bytes from 
     material and surface libs. 

The boundary and optical buffers are regarded as dynamic 
(although they may still be persisted for debugging/record keeping)


Domain
---------

When using --finebndtex option GBndLib::load with constituents true
does domain interpolation.  This and probably many other things that 
only happen on load need to be repositioned for direct GLTF to GPU approach,
in order to make the functionality available without doing a load.

**/
 

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GBndLib : public GPropertyLib {
  public:
       static const plog::Severity LEVEL ; 
       static const GBndLib* INSTANCE ; 
       static const GBndLib* GetInstance() ; 
       static unsigned MaterialIndexFromLine( unsigned line ) ;
  public:
       enum {
               OMAT,
               OSUR,
               ISUR,
               IMAT
            };
  public:
       void save();
       static GBndLib* load(Opticks* ok, bool constituents=false);
  public:
       GBndLib(Opticks* ok);
       GBndLib(Opticks* ok, GMaterialLib* mlib, GSurfaceLib* slib);
  private:
       void init(); 
  public:
       unsigned int getNumBnd() const ;
       void closeConstituents();
  public:
       std::string description(const guint4& bnd) const ;
       std::string shortname(const guint4& bnd) const ;
       std::string shortname(unsigned boundary) const ;
       bool contains(const guint4& bnd) const ;
       unsigned index(const guint4& bnd) const ;
  public:
       // boundary index lookups
       guint4 getBnd(unsigned int boundary) const ;
  public:
       unsigned getOuterMaterial(unsigned boundary) const;
       unsigned getOuterSurface(unsigned boundary) const;
       unsigned getInnerSurface(unsigned boundary) const;
       unsigned getInnerMaterial(unsigned boundary) const ;
  public:
       const char* getOuterMaterialName(unsigned boundary) const;
       const char* getOuterSurfaceName(unsigned boundary) const ;
       const char* getInnerSurfaceName(unsigned boundary) const ;
       const char* getInnerMaterialName(unsigned boundary) const ;
  public:
       // spec is added, yielding a boundary index 
       const char* getOuterMaterialName(const char* spec);
       const char* getOuterSurfaceName(const char* spec);
       const char* getInnerSurfaceName(const char* spec);
       const char* getInnerMaterialName(const char* spec);
  public:
       guint4 parse( const char* spec, bool flip=false) const ;
       bool contains( const char* spec, bool flip=false) const ;


  public:
       // Bnd (guint4) are only added if not already present
       // char* adders convert names to indices using m_mlib, m_slib 
       unsigned int addBoundary( const char* spec, bool flip=false ) ;
       unsigned int addBoundary( const char* omat, const char* osur, const char* isur, const char* imat) ;
  private:
       friend class GBndLibTest ; 
       void add(const guint4& bnd);
       guint4 add(const char* spec, bool flip=false);
       guint4 add(const char* omat, const char* osur, const char* isur, const char* imat);
       guint4 add(unsigned int omat, unsigned int osur, unsigned int isur, unsigned int imat);


  public:
       void loadIndexBuffer();
       void importIndexBuffer();
  public:
       void saveIndexBuffer();
       void saveOpticalBuffer();
       void dumpOpticalBuffer() const ;
  public:
       void saveAllOverride(const char* dir="$TMP");
  public:
       NPY<unsigned int>* createIndexBuffer();
       NPY<unsigned int>* createOpticalBuffer();
  public:
       bool hasIndexBuffer();
       NPY<unsigned int>* getIndexBuffer();
       NPY<unsigned int>* getOpticalBuffer();
  public:
       void setIndexBuffer(NPY<unsigned int>* index_buffer);
       void setOpticalBuffer(NPY<unsigned int>* optical_buffer);
  public:
       const std::map<std::string, unsigned>& getMaterialLineMap();
       const std::map<std::string, unsigned>& getMaterialLineMapConst() const ;
       void fillMaterialLineMap();
       void dumpMaterialLineMap(const char* msg="GBndLib::dumpMaterialLineMap"); 
  private:
       void fillMaterialLineMap(std::map<std::string, unsigned int>& msu);
       void dumpMaterialLineMap(std::map<std::string, unsigned int>& msu, const char* msg="GBndLib::dumpMaterialLineMap");
  public:
       unsigned int getMaterialLine(const char* shortname);
       static unsigned int getLine(unsigned int ibnd, unsigned int iquad);
       unsigned getMaterialIndexFromLine(unsigned line) const ;
 
       unsigned int getLineMin();
       unsigned int getLineMax();
  public:
       void createDynamicBuffers();
   public:
       // memcpy zip of materials and surfaces for each boundary from the mlib and slib buffers
       NPY<float>* createBufferForTex2d();  
       NPY<float>* createBufferOld();
  public:
       GItemList* createNames(); // spec shortnames
       NMeta*      createMeta();
       NPY<float>* createBuffer();  // invokes createBufferForTex2d
       void import();
       void sort();
       void defineDefaults(GPropertyMap<float>* defaults);
  public:
       void Summary(const char* msg="GBndLib::Summary");
       void dump(const char* msg="GBndLib::dump");
       void dumpBoundaries(std::vector<unsigned int>& boundaries, const char* msg="GBndLib::dumpBoundaries");
  public:
       void setMaterialLib(GMaterialLib* mlib);
       void setSurfaceLib(GSurfaceLib* slib);
       GMaterialLib* getMaterialLib();
       GSurfaceLib*  getSurfaceLib();
       bool isDbgBnd() const ; 
  private:
       bool                 m_dbgbnd ; 
       GMaterialLib*        m_mlib ; 
       GSurfaceLib*         m_slib ; 
       std::vector<guint4>  m_bnd ; 

       NPY<unsigned int>*   m_index_buffer ;  
       NPY<unsigned int>*   m_optical_buffer ;  
       std::map<std::string, unsigned int> m_materialLineMap ;

};

#include "GGEO_TAIL.hh"




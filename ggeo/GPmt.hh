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

struct NSlice ; 
class Opticks ; 

struct gbbox ; 
class GParts ; 
class GCSG ; 
class GBndLib ; 

/**

GPmt
======

Analytic PMT description obtained from parsing DetDesc
see python scripts in ~/opticks/ana/pmt (formerly ~/env/nuwa/detdesc/pmt/)
and pmt- bash functions 

This is mostly succeeded by full CSG tree, however still trying to 
keeping this alive for tpmt- and as the manually partitioned geometry 
is really fast to raytrace.

**/


#include "GGEO_API_EXPORT.hh"
class GGEO_API GPmt {
  public:
       static const char* FILENAME ;  
       static const char* FILENAME_CSG ;  
       static const char* GPMT ;  
       static const unsigned NTRAN ;  
   public:
       // loads persisted GParts buffer and associates with the GPmt
       static GPmt* load(Opticks* cache, GBndLib* bndlib, unsigned int index, NSlice* slice=NULL);
   public:
       GPmt(Opticks* cache, GBndLib* bndlib, unsigned int index);
       void setPath(const char* path);
       void dump(const char* msg="GPmt::dump");
   public:
       void addContainer(gbbox& bb, const char* bnd );
   private:
       void loadFromCache(NSlice* slice);    
       void setParts(GParts* parts);
       void setCSG(GCSG* csg);
   public:
       GParts*     getParts();
       GCSG*       getCSG();
       const char* getPath();
       unsigned    getIndex(); 
   private:
       Opticks*           m_ok ; 
       GBndLib*           m_bndlib ; 
       unsigned int       m_index ;
       GParts*            m_parts ;
       GCSG*              m_csg ;
       const char*        m_path ;
};



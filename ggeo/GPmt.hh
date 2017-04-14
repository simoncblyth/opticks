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
   public:
       void addContainer(gbbox& bb, const char* bnd );
   private:
       void loadFromCache(NSlice* slice);    
       void setParts(GParts* parts);
       void setCSG(GCSG* csg);
 
   public:
       GParts* getParts();
       GCSG*   getCSG();
       const char* getPath();
       unsigned    getIndex(); 
   private:
       Opticks*           m_cache ; 
       GBndLib*           m_bndlib ; 
       unsigned int       m_index ;
       GParts*            m_parts ;
       GCSG*              m_csg ;
       const char*        m_path ;
};




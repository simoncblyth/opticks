#pragma once
/**
CSG_stree_Convert.h : developing in CSG/tests for faster cycle 
================================================================

WARNING : WORKING ON THIS ELSEWHERE : LOOKS ALMOST COMPLETE THERE::

   CSGFoundry::importTree 
   CSGImport::importTree  

TODO: review progress in the approaches, consolidate, remove one of them  

* probably this one will be removed
* but the idea of developing header only for fast dev is a good one 
  worth bring over to CSGImport 



Named in the pattern of CSG_GGeo_Convert.
Once the old workflow is a distant memory can rename 
this something nicer. 


Aims is to replace cg:CSG_GGeo_Convert and 
hence totally remove the need for GGeo and a huge 
amount of code. Including the following packages::

    BRAP
    NPY
    OKC
    GGeo
    CSG_GGeo
    GeoChain 

**/

struct CSGFoundry ; 
struct stree ; 

struct CSG_stree_Convert
{
    static CSGFoundry* Translate(const stree* st );   

    CSGFoundry*  fd ;
    const stree* st  ;

    CSG_stree_Convert(CSGFoundry* fd, const stree* st ) ;
    void init();  
};

#include "CSGFoundry.h"
#include "stree.h"

inline CSGFoundry* CSG_stree_Convert::Translate(const stree* st)  // static 
{
    CSGFoundry* fd = new CSGFoundry ; 
    CSG_stree_Convert conv( fd, st ); 
    return fd ; 
}

inline CSG_stree_Convert::CSG_stree_Convert(CSGFoundry* fd_, const stree* st_ )
    :
    fd(fd_),
    st(st_)
{
    init(); 
}

/**
CSG_stree_Convert::init
-------------------------

Get started by following CSG_GGeo_Convert.cc swapping usage of 
GGeo API for equivalents in stree.h API

**/

#include "SGeoConfig.hh"
#include "SName.h"

inline void CSG_stree_Convert::init()
{
    st->get_meshname( fd->meshname ); 
    st->get_mmlabel( fd->mmlabel );     
    SGeoConfig::GeometrySpecificSetup(fd->id); 
    // SName takes a reference to the meshname vector of strings 

    const char* cxskiplv = SGeoConfig::CXSkipLV() ; 
    const char* cxskiplv_idxlist = SGeoConfig::CXSkipLV_IDXList() ;  
    fd->setMeta<std::string>("cxskiplv", cxskiplv ? cxskiplv : "-" ); 
    fd->setMeta<std::string>("cxskiplv_idxlist", cxskiplv_idxlist ? cxskiplv_idxlist : "-" ); 
    std::cerr 
        << " cxskiplv  " << cxskiplv 
        << " cxskiplv   " << cxskiplv
        << " fd.meshname.size " << fd->meshname.size()
        << " fd.id.getNumName " << fd->id->getNumName()
        << std::endl     
        ;    
}



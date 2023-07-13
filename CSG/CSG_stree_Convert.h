#pragma once
/**
CSG_stree_Convert
===================

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

inline void CSG_stree_Convert::init()
{
}





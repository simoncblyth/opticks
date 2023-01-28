#pragma once
/**
CSGTarget.h : const CSGFoundry ctor argument, sframe/CE:center_extent/transform access
========================================================================================

* provides CSGFoundry::target instance budding off transform related access 

::

    epsilon:CSG blyth$ opticks-f CSGTarget.h 
    ./CSG/CMakeLists.txt:    CSGTarget.h
    ./CSG/CSGTarget.cc:#include "CSGTarget.h"
    ./CSG/CSGTarget.h:CSGTarget.h : const CSGFoundry ctor argument, sframe/CE:center_extent/transform access
    ./CSG/CSGFoundry.cc:#include "CSGTarget.h"

**/

#include "plog/Severity.h"

struct CSGFoundry ; 
struct qat4 ; 
struct sframe ; 

struct CSGTarget
{
    static const plog::Severity LEVEL ; 
    const CSGFoundry* foundry ; 

    CSGTarget( const CSGFoundry* foundry );  


    int getFrame(sframe& fr,  int midx, int mord, int iidxg ) const ; 

    int getCenterExtent(float4& ce, int midx, int mord, int iidx=-1, qat4* m2w=nullptr, qat4* w2m=nullptr ) const ;

    int getFrame(sframe& fr,  int inst_idx ) const ; 

    int getLocalCenterExtent( float4& lce, int midx, int mord) const ;
    int getGlobalCenterExtent(float4& gce, int midx, int mord, int iidx, qat4* m2w=nullptr, qat4* w2m=nullptr ) const ; 

    int getTransform(qat4& q, int midx, int mord, int iidx) const  ; 
    const qat4* getInstanceTransform(int midx, int mord, int iidx) const ; 


};



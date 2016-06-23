#pragma once

class GMergedMesh ; 

#include "GMesh.hh"
#include "GGEO_API_EXPORT.hh"
class GGEO_API GBBoxMesh : public GMesh {
public:
    //enum { NUM_VERTICES = 8, NUM_FACES = 6*2 } ;
    enum { NUM_VERTICES = 24, NUM_FACES = 6*2 } ;
public:
    static GBBoxMesh* create(GMergedMesh* mergedmesh);
private:
    void eight(); 
    void twentyfour(); 
public:
    static void twentyfour(gbbox& bb, gfloat3* vertices, guint3* faces, gfloat3* normals);
public:
    GBBoxMesh(GMergedMesh* mm) ; 
    virtual ~GBBoxMesh(); 
private:
    GMergedMesh* m_mergedmesh ; 
     
};




#ifndef GMERGEDMESH_H
#define GMERGEDMESH_H


class GGeo ; 
class GNode ;
#include "GMesh.hh"

class GMergedMesh : public GMesh {
   
public:
    enum { pass_count, pass_merge } ;

    static GMergedMesh* create(unsigned int index, GGeo* ggeo);
    static GMergedMesh* load(const char* dir);

    GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;
    virtual ~GMergedMesh(); 

    void traverse( GNode* node, unsigned int depth, unsigned int pass);

    float* getModelToWorldPtr(unsigned int index);

    void dumpSolids(const char* msg="GMergedMesh::dumpSolids");
    void dumpWavelengthBuffer(unsigned int numBoundary, unsigned int numProp, unsigned int numSamples);

private:
     // transients that do not need persisting
     // keeping things needing persisting down in GMesh
     unsigned int m_cur_vertices ;
     unsigned int m_cur_faces ;
     unsigned int m_cur_solid ;
};




#endif

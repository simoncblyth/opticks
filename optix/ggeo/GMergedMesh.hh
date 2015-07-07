#pragma once

class GGeo ; 
class GNode ;

#include "GMesh.hh"

class GMergedMesh : public GMesh {
   
public:
    enum { pass_count, pass_merge } ;

public:
    static GMergedMesh* create(unsigned int index, GGeo* ggeo);
    static GMergedMesh* load(const char* dir);

public:
    GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;
    virtual ~GMergedMesh(); 

private:
    void traverse( GNode* node, unsigned int depth, unsigned int pass);

public:
    float* getModelToWorldPtr(unsigned int index);
    void dumpSolids(const char* msg="GMergedMesh::dumpSolids");
    void dumpWavelengthBuffer(unsigned int numBoundary, unsigned int numProp, unsigned int numSamples);
    gfloat3* getNodeColor(GNode* node);

private:
    // transients that do not need persisting
    // keeping things needing persisting down in GMesh
    unsigned int m_cur_vertices ;
    unsigned int m_cur_faces ;
    unsigned int m_cur_solid ;
     
};


inline GMergedMesh::GMergedMesh(GMergedMesh* other)
       : 
       GMesh(other),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0)
{
}

inline GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0)
{
} 

inline GMergedMesh::~GMergedMesh()
{
}





#pragma once

#include <map>
#include <vector>
#include <string>

class GGeo ; 
class GNode ;

#include "GMesh.hh"

//
// NB : GMergedMesh is just relevant for GMesh creation from multiple GMesh 
//      general usage should target GMesh  
//
// NB : DO NOT ADD METHODS HERE THAT CAN LIVE IN GMesh
//

class GMergedMesh : public GMesh {
   
public:
    enum { PASS_COUNT, PASS_MERGE } ;

public:
    static GMergedMesh* create(unsigned int index, GGeo* ggeo, GNode* base=NULL);
    static GMergedMesh* load(const char* dir, unsigned int index=0, const char* version=NULL );

public:
    GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;

    virtual ~GMergedMesh(); 
    void reportMeshUsage(GGeo* ggeo, const char* msg="GMergedMesh::reportMeshUsage");

private:
    void traverse( GNode* node, unsigned int depth, unsigned int pass);

public:
    float* getModelToWorldPtr(unsigned int index);
    void dumpSolids(const char* msg="GMergedMesh::dumpSolids");
    void dumpWavelengthBuffer(unsigned int numBoundary, unsigned int numProp, unsigned int numSamples);
    gfloat3* getNodeColor(GNode* node);

public:
    // used when obtaining relative transforms for flattening sub-trees of repeated geometry
    void   setCurrentBase(GNode* base);
    GNode* getCurrentBase(); 
    bool   isGlobal(); 
    bool   isInstanced(); 

private:
    // transients that do not need persisting
    // keeping things needing persisting down in GMesh
    unsigned int m_cur_vertices ;
    unsigned int m_cur_faces ;
    unsigned int m_cur_solid ;
    GNode*       m_cur_base ;  

    std::map<unsigned int, unsigned int> m_mesh_usage ; 
     
};


inline GMergedMesh::GMergedMesh(GMergedMesh* other)
       : 
       GMesh(other),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0),
       m_cur_base(NULL)
{
}

inline GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0),
       m_cur_base(NULL)
{
} 

inline GMergedMesh::~GMergedMesh()
{
}

inline void GMergedMesh::setCurrentBase(GNode* base)
{
    m_cur_base = base ; 
}
inline GNode* GMergedMesh::getCurrentBase()
{
    return m_cur_base ; 
}
inline bool GMergedMesh::isGlobal()
{
    return m_cur_base == NULL ; 
}
inline bool GMergedMesh::isInstanced()
{
    return m_cur_base != NULL ; 
}




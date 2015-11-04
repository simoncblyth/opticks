#pragma once

// *GMergedMesh* 
//     is just relevant for GMesh creation from multiple GMesh general usage should target GMesh  
//
//     THAT MEANS : DO NOT ADD METHODS HERE THAT CAN LIVE IN GMesh

#include <map>
#include <vector>
#include <string>

class GCache ; 
class GGeo ; 
class GNode ;
class GSolid ; 

#include "GMesh.hh"
#include "GVector.hh"

class GMergedMesh : public GMesh {
public:
    enum { PASS_COUNT, PASS_MERGE } ;
public:
    static GMergedMesh* create(unsigned int index, GGeo* ggeo, GNode* base=NULL);
    static GMergedMesh* load(GCache* cache  , unsigned int index=0, const char* version=NULL );
    static GMergedMesh* load(const char* dir, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, std::vector<GSolid*>& solids) ;
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, GSolid* solid ) ;
public:
    //GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;
private:
    // NB cannot treat GMergedMesh as a GMesh wrt calling getNumSolids 
    // explicit naming to avoid subclass confusion
    void countMergedMesh( GMergedMesh* other, bool selected );   
    void countSolid( GSolid*      solid, bool selected ); 
    void countMesh( GMesh* mesh ); 
    void mergeSolid( GSolid* solid, bool selected );
    void mergeMergedMesh( GMergedMesh* other, bool selected );
public:
    void traverse( GNode* node, unsigned int depth, unsigned int pass);
public:
    float* getModelToWorldPtr(unsigned int index);
    void reportMeshUsage(GGeo* ggeo, const char* msg="GMergedMesh::reportMeshUsage");
    void dumpSolids(const char* msg="GMergedMesh::dumpSolids");
public:
    // used when obtaining relative transforms for flattening sub-trees of repeated geometry
    void   setCurrentBase(GNode* base);
    GNode* getCurrentBase(); 
    bool   isGlobal(); 
    bool   isInstanced(); 
private:
    // transients that do not need persisting, persistables are down in GMesh
    unsigned int m_cur_vertices ;
    unsigned int m_cur_faces ;
    unsigned int m_cur_solid ;
    GNode*       m_cur_base ;  
    std::map<unsigned int, unsigned int> m_mesh_usage ; 
     
};


/*
inline GMergedMesh::GMergedMesh(GMergedMesh* other)
       : 
       GMesh(other),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0),
       m_cur_base(NULL)
{
}
*/

inline GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0),
       m_cur_base(NULL)
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



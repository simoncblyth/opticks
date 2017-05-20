#pragma once

// *GMergedMesh* 
//     is just relevant for GMesh creation from multiple GMesh general usage should target GMesh  
//
//     THAT MEANS : DO NOT ADD METHODS HERE THAT CAN LIVE IN GMesh

#include <map>
#include <vector>
#include <string>

class Opticks ; 

class GGeo ; 
class GNode ;
class GSolid ; 
class GParts ; 
class GMergedMesh ; 

#include "GMesh.hh"
#include "GVector.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"


class GGEO_API GMergedMesh : public GMesh {
    friend class GGeoTest ;         // for setParts analytic PMT kludge
    friend class OpticksGeometry ;  // for setParts analytic PMT kludge
public:
    enum { PASS_COUNT, PASS_MERGE } ;
public:
    static GMergedMesh* create(unsigned ridx, GNode* base, GNode* root, unsigned verbosity);
private:
     // operates in COUNT and MERGE passes, COUNT find out the number of 
     // ridx selected solids and their vertices to allocate then 
     // MERGE collects them together
     void traverse_r( GNode* node, unsigned int depth, unsigned int pass, unsigned verbosity );

public:
    static GMergedMesh* load(Opticks* opticks, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* load(const char* dir, unsigned int index=0, const char* version=NULL );
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, const std::vector<GSolid*>& solids, unsigned verbosity) ;
    static GMergedMesh* combine(unsigned int index, GMergedMesh* mm, GSolid* solid, unsigned verbosity ) ;
public:
    //GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;
    GParts* getParts();
private:
    void setParts(GParts* pts); 
private:
    // NB cannot treat GMergedMesh as a GMesh wrt calling getNumSolids 
    // explicit naming to avoid subclass confusion
    void countMergedMesh( GMergedMesh* other, bool selected );   
    void countSolid( GSolid*      solid, bool selected, unsigned verbosity  ); 
    void countMesh( GMesh* mesh ); 
    void mergeSolid( GSolid* solid, bool selected, unsigned verbosity );
    void mergeMergedMesh( GMergedMesh* other, bool selected );

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
public:
    // geocodes used to communicate between ggv- oglrap- optixrap-
    bool   isSkip(); 
    bool   isAnalytic(); 
    bool   isTriangulated(); 
private:
    // transients that do not need persisting, persistables are down in GMesh
    unsigned int m_cur_vertices ;
    unsigned int m_cur_faces ;
    unsigned int m_cur_solid ;
    GNode*       m_cur_base ;  
    GParts*      m_parts ; 
    std::map<unsigned int, unsigned int> m_mesh_usage ; 
     
};

#include "GGEO_TAIL.hh"


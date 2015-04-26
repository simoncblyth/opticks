#ifndef GMERGEDMESH_H
#define GMERGEDMESH_H

class GGeo ; 
class GNode ;
#include "GMesh.hh"

class GMergedMesh : public GMesh {
   
public:
    enum { pass_count, pass_merge } ;

    static GMergedMesh* create(unsigned int index, GGeo* ggeo);

    GMergedMesh(GMergedMesh* other) ;  // stealing copy ctor
    GMergedMesh(unsigned int index) ;
    virtual ~GMergedMesh(); 
    void traverse( GNode* node, unsigned int depth, unsigned int pass);


    float* getModelToWorldPtr(unsigned int index);
    gfloat4 getCenterExtent(unsigned int index);


    unsigned int getNumSolids();
    unsigned int getNumSolidsSelected();
    void dumpSolids(const char* msg="GMergedMesh::dumpSolids");

private:
     unsigned int m_cur_vertices ;
     unsigned int m_cur_faces ;
     unsigned int m_cur_solid ;
     unsigned int m_num_solids  ;
     unsigned int m_num_solids_selected  ;


};


inline unsigned int GMergedMesh::getNumSolids()
{
    return m_num_solids ; 
}
inline unsigned int GMergedMesh::getNumSolidsSelected()
{
    return m_num_solids_selected ; 
}



#endif

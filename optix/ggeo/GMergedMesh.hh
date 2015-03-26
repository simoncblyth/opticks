#ifndef GMERGEDMESH_H
#define GMERGEDMESH_H

class GGeo ; 
class GNode ;
#include "GMesh.hh"

class GMergedMesh : public GMesh {
   
public:
    enum { pass_count, pass_merge } ;

    static GMergedMesh* create(unsigned int index, GGeo* ggeo);

    GMergedMesh(unsigned int index) ;
    virtual ~GMergedMesh(); 
    void traverse( GNode* node, unsigned int depth, unsigned int pass);

private:
     unsigned int m_cur_vertices ;
     unsigned int m_cur_faces ;


};

#endif

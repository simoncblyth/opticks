#ifndef GGEOINTERCONNECT_H
#define GGEOINTERCONNECT_H

#include "IGeometry.hh"

class Buffer ; 
class GMergedMesh ;

// T-connector avoiding ggeo having to depend on OGLRap/IGeometry
class GGeoInterConnect : public IGeometry {
    public:
       GGeoInterConnect(GMergedMesh* mm);
       virtual ~GGeoInterConnect();

   public:
       unsigned int getNumElements();
       Buffer* getVertices();
       Buffer* getColors();
       Buffer* getIndices();
       void Summary(const char* msg="GGeoInterConnect::Summary");

   private:
       unsigned int m_num_elements ;
       Buffer* m_vertices ;
       Buffer* m_colors ;
       Buffer* m_indices  ;

};

#endif

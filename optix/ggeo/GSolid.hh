#pragma once

class GMesh ;
class GBoundary ; 
//class GOpticalSurface ;

#include "GNode.hh"
#include "GMatrix.hh"

//  hmm the difference between the models is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//

class GSolid : public GNode {
  public:
      GSolid( unsigned int index, GMatrixF* transform, GMesh* mesh,  GBoundary* boundary);
      virtual ~GSolid();

  public:
     void setSelected(bool selected);
     bool isSelected();

  public:
     void setBoundary(GBoundary* boundary);

  public:
     GBoundary* getBoundary();

  public: 
      void Summary(const char* msg="GSolid::Summary");
 
  private:
      GBoundary*        m_boundary ; 
      bool m_selected ;

};

inline GSolid::GSolid( unsigned int index, GMatrixF* transform, GMesh* mesh, GBoundary* boundary)
         : 
         GNode(index, transform, mesh ),
         m_boundary(boundary),
         m_selected(true)
{
    // NB not taking ownership yet 
}

inline GSolid::~GSolid()
{
}

inline GBoundary* GSolid::getBoundary()
{
    return m_boundary ; 
}



inline void GSolid::setSelected(bool selected)
{
    m_selected = selected ; 
}
inline bool GSolid::isSelected()
{
   return m_selected ; 
}






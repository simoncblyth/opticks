#ifndef GSOLID_H
#define GSOLID_H

class GMesh ;
class GSubstance ; 

#include "GNode.hh"
#include "GMatrix.hh"

//  hmm the difference between the models is focussed in here 
//   chroma.geometry.Solid is all about splaying things across all the triangles
//  relationship between how many materials for each mesh is up for grabs
//

class GSolid : public GNode {
  public:
      GSolid( unsigned int index, GMatrixF* transform, GMesh* mesh,  GSubstance* substance);
      virtual ~GSolid();

  public:
     void setSelected(bool selected);
     bool isSelected();

  public:
     void setSubstance(GSubstance* substance);
     GSubstance* getSubstance();

  public: 
      void Summary(const char* msg="GSolid::Summary");
 
  private:
      GSubstance* m_substance ; 
      bool m_selected ;

};


#endif

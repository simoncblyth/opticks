#ifndef GNODE_H
#define GNODE_H

#include <vector>

#include "GVector.hh"
#include "GMatrix.hh"
class GMesh ;

class GNode {
  public:
      GNode(unsigned int index, GMatrixF* transform, GMesh* mesh);
      virtual ~GNode();

  public: 
      void Summary(const char* msg="GNode::Summary");

  public:
      void setParent(GNode* parent);
      void addChild(GNode* child);
      void setDescription(char* desc);

  public:
      unsigned int getIndex();
      GNode* getParent(); 
      GNode* getChild(unsigned int index);
      unsigned int getNumChildren();
      char* getDescription();

  public:
     void updateBounds();
     void updateBounds(gfloat3& low, gfloat3& high );

     gfloat3* getLow();
     gfloat3* getHigh();

  public:
     GMesh* getMesh();
     GMatrixF* getTransform();

  private:
      unsigned int m_index ; 
      GNode* m_parent ; 
      std::vector<GNode*> m_children ;
      char* m_description ;

  private: 
      GMatrixF* m_transform ; 
      GMesh*     m_mesh ; 
      gfloat3*   m_low ; 
      gfloat3*   m_high ; 
 

};


#endif

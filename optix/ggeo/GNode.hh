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
      //
      // **boundary indices live on the node rather than the mesh**
      //
      // as there are a relatively small number of meshes and many nodes
      // that utilize them with different transforms
      //
      // normally a single boundary per-node but allow the 
      // possibility of compound boundary nodes, eg for combined meshes
      //
      void setBoundaryIndices(unsigned int boundary_index);
      void setBoundaryIndices(unsigned int* boundary_indices);

      unsigned int* getBoundaryIndices();
      std::vector<unsigned int>& getDistinctBoundaryIndices();
      void updateDistinctBoundaryIndices();

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
      unsigned int* getNodeIndices();
  private:
      void setNodeIndices(unsigned int index); 

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

  private: 
      unsigned int* m_boundary_indices ;
      unsigned int* m_node_indices ;

  private: 
      std::vector<unsigned int> m_distinct_boundary_indices ;

};


#endif

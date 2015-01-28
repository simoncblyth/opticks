#ifndef ASSIMPNODE_H
#define ASSIMPNODE_H

#include <vector>
#include <assimp/types.h>

class AssimpTree ; 
struct aiNode ; 
struct aiMesh ; 

class AssimpNode {
      friend class AssimpTree ; 

  public:
      AssimpNode(aiNode* node, AssimpTree* tree);
      virtual ~AssimpNode();

  public:
      void setIndex(unsigned int index);
      void setDepth(unsigned int depth);
      void setParent(AssimpNode* parent);
      void addChild(AssimpNode* child);
      void traverse();
      void dump();

  public:
      void summary(const char* msg="AssimpNode::summary");
      void bounds(const char* msg="AssimpNode::bounds");
      aiVector3D* getLow();
      aiVector3D* getHigh();
      aiVector3D* getCenter();

      void updateBounds();
      void updateBounds(aiVector3D& low, aiVector3D& high);


  public:
      const char* getName();
      AssimpNode* getParent();
      unsigned int getNumChildren();
      AssimpNode* getChild(unsigned int n);
      unsigned int getIndex();
      unsigned int getDepth();

  public:
      void ancestors(); 
      unsigned int progeny(); 

  public:
      void copyMeshes(aiMatrix4x4 transform);
      aiMatrix4x4 getTransform();
      unsigned int getNumMeshes();
      unsigned int getNumMeshesRaw();
      unsigned int getMeshIndexRaw(unsigned int index);
      aiMesh* getRawMesh(unsigned int index);
      aiMesh* getMesh(unsigned int index);

  protected: 
      aiNode* getRawNode();

  private:
      unsigned int m_index ; 

      unsigned int m_depth ; 

      aiNode* m_raw ; 

      aiMesh** m_meshes; 

      unsigned int m_numMeshes ;

      aiMatrix4x4 m_transform ; 

      aiVector3D* m_low ; 

      aiVector3D* m_high ; 

      aiVector3D* m_center ; 

      AssimpTree* m_tree ; 

      AssimpNode* m_parent ; 

      std::vector<AssimpNode*> m_children ;

};


#endif



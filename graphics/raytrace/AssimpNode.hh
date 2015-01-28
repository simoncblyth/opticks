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
      void setTransform(aiMatrix4x4 transform);
      void addChild(AssimpNode* child);
      void traverse();
      void dump();

  public:
      AssimpNode* getParent();
      unsigned int getNumChildren();
      AssimpNode* getChild(unsigned int n);
      unsigned int getIndex();
      unsigned int getDepth();
      aiMatrix4x4 getTransform();

      const char* getName();
      void ancestors(); 
      unsigned int progeny(); 

  public:
      unsigned int getNumMeshes();
      unsigned int getMeshIndex(unsigned int index);
      aiMesh* getRawMesh(unsigned int index);

  protected: 
      aiNode* getRawNode();

  private:
      unsigned int m_index ; 

      unsigned int m_depth ; 

      aiNode* m_raw ; 

      aiMatrix4x4 m_transform ; 

      AssimpTree* m_tree ; 

      AssimpNode* m_parent ; 

      std::vector<AssimpNode*> m_children ;

};


#endif



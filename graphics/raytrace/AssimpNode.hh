#ifndef ASSIMPNODE_H
#define ASSIMPNODE_H

#include <vector>
#include <assimp/types.h>

struct aiNode ; 

class AssimpNode {
      friend class AssimpTree ; 

  public:
      AssimpNode(aiNode* node);
      virtual ~AssimpNode();

  public:
      void setIndex(unsigned int index);
      void setDepth(unsigned int depth);

      void setParent(AssimpNode* parent);
      void setTransform(aiMatrix4x4 transform);
      void addChild(AssimpNode* child);
      void traverse(AssimpNode* node=NULL);
      void dump();

  public:
      AssimpNode* getParent();
      unsigned int getNumChildren();
      AssimpNode* getChild(unsigned int n);
      unsigned int getIndex();
      unsigned int getDepth();
      aiMatrix4x4 getTransform();

  protected: 
      aiNode* getRawNode();

  private:
      unsigned int m_index ; 

      unsigned int m_depth ; 

      aiNode* m_rawnode ; 

      aiMatrix4x4 m_transform ; 

      AssimpNode* m_parent ; 

      std::vector<AssimpNode*> m_children ;

};


#endif



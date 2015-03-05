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
      AssimpNode(std::vector<aiNode*> nodepath, AssimpTree* tree);
      virtual ~AssimpNode();

  public:
      void setIndex(unsigned int index);
      void setDepth(unsigned int depth);
      void setParent(AssimpNode* parent);
      void addChild(AssimpNode* child);
      void traverse();
      void dump();

  public: 
      aiMatrix4x4 getGlobalTransform();

  public:
      void summary(const char* msg="AssimpNode::summary");
      void bounds(const char* msg="AssimpNode::bounds");
      aiVector3D* getLow();
      aiVector3D* getHigh();
      aiVector3D* getCenter();
      aiVector3D* getExtent();

      void updateBounds();
      void updateBounds(aiVector3D& low, aiVector3D& high);


  public:
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
      //unsigned int getMeshIndexRaw(unsigned int index);
      aiMesh* getMesh(unsigned int index);

  public:
      aiMesh* getRawMesh(unsigned int localMeshindex=0);
      unsigned int getMaterialIndex(unsigned int localMeshIndex=0);
      unsigned int getMeshIndex(unsigned int localMeshIndex=0);

  public:
      std::size_t getDigest();
      std::size_t getParentDigest();
  protected:
      std::size_t hash(unsigned int pyfirst, unsigned int pylast);

  public: 
      aiNode* getRawNode();
      const char* getName();
      aiNode* getRawNode(unsigned int iback);
      const char* getName(unsigned int iback);

  private:
      unsigned int m_index ; 

      unsigned int m_depth ; 

      aiNode* m_raw ; 

      std::vector<aiNode*> m_nodepath ; 

      std::size_t m_digest ;

      std::size_t m_pdigest ;

      aiMesh** m_meshes; 

      unsigned int m_numMeshes ;

      aiMatrix4x4 m_transform ; 

      aiVector3D* m_low ; 
      aiVector3D* m_high ; 
      aiVector3D* m_center ; 
      aiVector3D* m_extent ; 

      AssimpTree* m_tree ; 

      AssimpNode* m_parent ; 

      std::vector<AssimpNode*> m_children ;

};


#endif



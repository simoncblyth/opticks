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
      void setParent(AssimpNode* parent);
      void addChild(AssimpNode* child);
      void setIndex(unsigned int index);
      void setDepth(unsigned int depth);

  public:
      AssimpNode* getParent();
      AssimpNode* getChild(unsigned int n);
      unsigned int getNumChildren();
      unsigned int getIndex();
      unsigned int getDepth();

  public:
      void ancestors(); 
      unsigned int progeny(); 

  public:
      void summary(const char* msg="AssimpNode::summary");
      void traverse();
      void dump();

  public: 
      aiMatrix4x4 getGlobalTransform();

  public:
      void bounds(const char* msg="AssimpNode::bounds");
      void updateBounds();
      void updateBounds(aiVector3D& low, aiVector3D& high);

      aiVector3D* getLow();
      aiVector3D* getHigh();
      aiVector3D* getCenter();
      aiVector3D* getExtent();

  public:
      void copyMeshes(aiMatrix4x4 transform);
      aiMatrix4x4 getTransform();
      unsigned int getNumMeshes();
      unsigned int getNumMeshesRaw();
      aiMesh* getMesh(unsigned int index);

  public:
      aiMesh* getRawMesh(unsigned int localMeshindex=0);
      unsigned int getMeshIndex(unsigned int localMeshIndex=0);
      unsigned int getMaterialIndex(unsigned int localMeshIndex=0);
      char* getMaterialName(unsigned int localMeshIndex=0);
      char* getDescription(const char* label="AssimpNode::getDescription"); // caller should free

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

  private:
      aiMesh** m_meshes; 
      unsigned int m_numMeshes ;
      aiMatrix4x4 m_transform ; 

  private:
      aiVector3D* m_low ; 
      aiVector3D* m_high ; 
      aiVector3D* m_center ; 
      aiVector3D* m_extent ; 

  private:
      AssimpTree* m_tree ; 
      AssimpNode* m_parent ; 
      std::vector<AssimpNode*> m_children ;

};


#endif



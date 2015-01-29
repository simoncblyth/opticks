#ifndef ASSIMPTREE_H
#define ASSIMPTREE_H

#include <assimp/types.h>
#include <vector>

class AssimpNode ; 
struct aiScene ; 
struct aiMesh ; 
struct aiNode ; 

// strings together a tree of AssimpNode

class AssimpTree {

public:
    AssimpTree(const aiScene* scene);
    virtual ~AssimpTree();

public:
    AssimpNode* getRoot();
    void setRoot(AssimpNode* root);

    aiMesh* getRawMesh(unsigned int meshIndex );

    void traverse();
    void dump();

public:
    unsigned int select(const char* query);
    void dumpSelection();
    unsigned int getNumSelected();
    AssimpNode* getSelectedNode(unsigned int i);

public:
    // bounds
    void bounds();
    aiVector3D* getLow();
    aiVector3D* getHigh();
    aiVector3D* getCenter();
    aiVector3D* getExtent();

    void findBounds();
    void findBounds(AssimpNode* node, aiVector3D& low, aiVector3D& high );

public:
    // examination of raw tree
    void traverseRaw(const char* msg="AssimpTree::traverseRaw");
    void traverseRaw(aiNode* raw, std::vector<aiNode*> ancestors);
    void visitRaw(aiNode* raw, std::vector<aiNode*> ancestors);

public:
    // wrapping raw tree matching pycollada/g4daenode 
    void traverseWrap(const char* msg="AssimpTree::traverseWrap");
    void traverseWrap(aiNode* node, std::vector<aiNode*> ancestors);

private:
   // initial wrap attempt 
   //void wrap(AssimpNode* node, unsigned int depth, aiMatrix4x4 transform);

private:
    void selectNodes(const char* query, AssimpNode* node, unsigned int depth);



private:

   std::vector<AssimpNode*> m_selection ; 

   unsigned int m_index ; 

   unsigned int m_raw_index ; 

   unsigned int m_wrap_index ; 

   unsigned int m_dump ; 

   const aiScene* m_scene ;  

   AssimpNode* m_root ;  

   char* m_query ; 

   aiVector3D* m_low ; 

   aiVector3D* m_high ; 

   aiVector3D* m_center ; 

   aiVector3D* m_extent ; 


};


#endif

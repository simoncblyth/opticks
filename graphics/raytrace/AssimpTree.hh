#ifndef ASSIMPTREE_H
#define ASSIMPTREE_H

#include <assimp/types.h>
#include <vector>

class AssimpNode ; 
struct aiScene ; 
struct aiMesh ; 

// strings together a tree of AssimpNode

class AssimpTree {

public:
    AssimpTree(const aiScene* scene);
    virtual ~AssimpTree();

public:
    AssimpNode* getRoot();

    aiMesh* getRawMesh(unsigned int meshIndex );

    void traverse();

    unsigned int select(const char* query);

    unsigned int getNumSelected();

    AssimpNode* getSelectedNode(unsigned int i);

private:

   void wrap(AssimpNode* node, unsigned int depth, aiMatrix4x4 transform);

private:

    void selectNodes(const char* query, AssimpNode* node, unsigned int depth );


private:

    AssimpNode* searchNode(const char* query);


private:

   std::vector<AssimpNode*> m_selection ; 

   unsigned int m_index ; 

   const aiScene* m_scene ;  

   AssimpNode* m_root ;  

};


#endif

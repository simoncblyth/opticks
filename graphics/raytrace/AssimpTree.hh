#ifndef ASSIMPTREE_H
#define ASSIMPTREE_H

class AssimpNode ; 

#include <assimp/types.h>

struct aiNode ; 

class AssimpTree {

public:
   AssimpTree(aiNode* top);
   virtual ~AssimpTree();
   AssimpNode* getRoot();

private:
   void traverse(AssimpNode* node, unsigned int depth, aiMatrix4x4 transform);

private:
   unsigned int m_index ; 

   AssimpNode* m_root ;  

};


#endif

#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include <stdio.h>
#include <assimp/scene.h>

AssimpTree::AssimpTree(aiNode* top) 
  : 
  m_root(NULL),
  m_index(0)
{
   m_root = new AssimpNode(top);
   m_root->setIndex(0);

   aiMatrix4x4 identity ; 
   traverse( m_root, 0, identity);
   printf("AssimpTree::AssimpTree created tree of %d AssimpNode \n", m_index );
}


AssimpNode* AssimpTree::getRoot()
{
   return m_root ; 
}

void AssimpTree::traverse(AssimpNode* node, unsigned int depth, aiMatrix4x4 accTransform)
{
   aiNode* raw = node->getRawNode(); 
   aiMatrix4x4 transform = raw->mTransformation * accTransform ;

   node->setIndex(m_index);
   node->setDepth(depth);
   node->setTransform(transform);

   m_index++ ; 

   //
   // accTransform     : accumulated transforms from parentage
   // raw->mTransform  : this node relative to parent 
   // transform        : this node global transform 
   //
   //printf("AssimpTree::traverse index %d depth %d \n", index, depth );

   for(unsigned int i = 0; i < raw->mNumChildren; i++) 
   {
       AssimpNode* child = new AssimpNode(raw->mChildren[i]);

       child->setParent(node);
       node->addChild(child);

       traverse(child, depth + 1, transform);
   }
}


AssimpTree::~AssimpTree()
{
}








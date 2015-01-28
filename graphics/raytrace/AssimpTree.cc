#include "AssimpTree.hh"
#include "AssimpNode.hh"
#include <stdio.h>
#include <assimp/scene.h>

AssimpTree::AssimpTree(const aiScene* scene) 
  : 
  m_scene(scene),
  m_root(NULL),
  m_index(0)
{
   aiNode* top = scene->mRootNode ; 
   m_root = new AssimpNode(top, this);
   m_root->setIndex(0);

   aiMatrix4x4 identity ; 
   wrap( m_root, 0, identity);
   printf("AssimpTree::AssimpTree created tree of %d AssimpNode \n", m_index );
}

AssimpTree::~AssimpTree()
{
}


aiMesh* AssimpTree::getRawMesh(unsigned int meshIndex )
{
    aiMesh* mesh = m_scene->mMeshes[meshIndex];
    return mesh ; 
}

AssimpNode* AssimpTree::getRoot()
{
   return m_root ; 
}

void AssimpTree::traverse()
{
   m_root->traverse();
}


void AssimpTree::wrap(AssimpNode* node, unsigned int depth, aiMatrix4x4 accTransform)
{
   // wrapping the tree

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
       AssimpNode* child = new AssimpNode(raw->mChildren[i], this);

       child->setParent(node);
       node->addChild(child);

       wrap(child, depth + 1, transform);
   }
}





unsigned int AssimpTree::getNumSelected()
{
    return m_selection.size();
}

AssimpNode* AssimpTree::getSelectedNode(unsigned int i)
{
    return i < m_selection.size() ? m_selection[i] : NULL ; 
}


unsigned int AssimpTree::select(const char* query)
{
    if(!m_root)
    {
        printf("AssimpTree::select ERROR tree not yet wrapped \n");
        return 0 ;
    } 
    m_index = 0 ; 
    m_selection.clear();
    selectNodes(query, m_root, 0 );
    printf("AssimpTree::select query %s matched %lu nodes \n", query, m_selection.size() ); 
    return m_selection.size();
}

void AssimpTree::selectNodes(const char* query, AssimpNode* node, unsigned int depth )
{
   m_index++ ; 
   const char* name = node->getName(); 
   if(strncmp(name,query,strlen(query)) == 0){
      m_selection.push_back(node); 
   }
   for(unsigned int i = 0; i < node->getNumChildren(); i++) {   
      selectNodes(query, node->getChild(i), depth + 1 );
   }   
}


AssimpNode* AssimpTree::searchNode(const char* query)
{
    // non-uniqueness of node names makes this a bit useless anyhow
    return NULL ; 
}





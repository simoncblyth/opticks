#include "AssimpNode.hh"
#include "AssimpTree.hh"
#include "AssimpCommon.hh"
#include "AssimpMesh.hh"
#include "assert.h"

#include <assimp/scene.h>


AssimpNode::AssimpNode(aiNode* node, AssimpTree* tree) 
   : 
   m_parent(NULL),
   m_tree(tree),
   m_raw(node),
   m_index(0) 
{
}

AssimpNode::~AssimpNode()
{
}

aiNode* AssimpNode::getRawNode(){
   return m_raw ; 
}
const char* AssimpNode::getName() {
    return m_raw->mName.C_Str();
}

void AssimpNode::setParent(AssimpNode* parent){
    m_parent = parent ;
}
void AssimpNode::setIndex(unsigned int index){
    m_index = index ; 
}
void AssimpNode::setDepth(unsigned int depth){
    m_depth = depth ; 
}
void AssimpNode::setTransform(aiMatrix4x4 transform){
    m_transform = transform ; 
}
void AssimpNode::addChild(AssimpNode* child)
{
    m_children.push_back(child); 
}



AssimpNode* AssimpNode::getParent(){
    return m_parent ;  
}
unsigned int AssimpNode::getIndex(){
    return m_index ; 
}
unsigned int AssimpNode::getDepth(){
    return m_depth ; 
}
aiMatrix4x4 AssimpNode::getTransform(){
   return m_transform ; 
}
unsigned int AssimpNode::getNumChildren(){
    return m_children.size(); 
}
AssimpNode* AssimpNode::getChild(unsigned int n){
    return n < getNumChildren() ? m_children[n] : NULL ;
}


void AssimpNode::dump()
{
    unsigned int nchild = getNumChildren();
    unsigned int nprog = progeny();
    unsigned int nmesh = getNumMeshes() ;

    if(nchild > 8)
    {
        printf("AssimpNode::dump index %5d depth %2d nchild %4d nprog %6d nmesh %d name %s  \n", m_index, m_depth, nchild, nprog, nmesh, getName() );
        for(unsigned int i = 0; i < nmesh ; i++)
        {   

            unsigned int meshIndex = m_raw->mMeshes[i];

            aiMesh* mesh = getRawMesh(i);

            printf("AssimpNode::dump  i %d meshIndex %d \n", i, meshIndex);

            dumpMesh(mesh);

            // mesh in global coordinates
            AssimpMesh* am = new AssimpMesh(mesh, m_transform);
            dumpMesh(am->getRawMesh());
        }   
    }
}




unsigned int AssimpNode::getNumMeshes()
{
    return m_raw->mNumMeshes ; 
}

unsigned int AssimpNode::getMeshIndex(unsigned int index)
{
    // node index to "global" scene mesh index
    return m_raw->mMeshes[index];
}

aiMesh* AssimpNode::getRawMesh(unsigned int index)
{
     unsigned int meshIndex = getMeshIndex(index);

     aiMesh* mesh = m_tree->getRawMesh(meshIndex);

     return mesh ;
}




void AssimpNode::traverse()
{
   dump();
   for(unsigned int i=0 ; i < getNumChildren() ; i++ ) getChild(i)->traverse(); 
}

void AssimpNode::ancestors()
{
   AssimpNode* node = this ; 
   unsigned int count = 0 ;
   while( node )
   {
       printf("AssimpNode::ancestors %2d %s \n", count, node->getName() );
       node = node->getParent();
       count++ ;
   }
}

unsigned int AssimpNode::progeny()
{
   //printf("AssimpNode::progeny %s \n", name());
   unsigned int nchild = getNumChildren();
   unsigned int tot = nchild ; 
   for(unsigned int i=0 ; i < nchild ; i++ ) tot += getChild(i)->progeny(); 
   return tot ; 
}







#include "AssimpNode.hh"
#include "assert.h"


AssimpNode::AssimpNode(aiNode* node) 
   : 
   m_parent(NULL),
   m_rawnode(node),
   m_index(0) 
{
}

AssimpNode::~AssimpNode()
{
}

aiNode* AssimpNode::getRawNode()
{
   return m_rawnode ; 
}

aiMatrix4x4 AssimpNode::getTransform()
{
   return m_transform ; 
}


void AssimpNode::dump()
{
    printf("AssimpNode::dump index %d depth %d #child %d \n", m_index, m_depth, getNumChildren() );
}


void AssimpNode::traverse(AssimpNode* node)
{
   assert(node);

   node->dump();

   for(unsigned int i=0 ; i < node->getNumChildren() ; i++ )
   {
       AssimpNode* child = node->getChild(i);
       traverse(child); 
   } 
}


void dumpTransform(aiMatrix4x4 t)
{
   printf("AssimpNode::dumpTransform\n");
   printf("a %10.4f %10.4f %10.4f %10.4f \n", t.a1, t.a2, t.a3, t.a4 );
   printf("b %10.4f %10.4f %10.4f %10.4f \n", t.b1, t.b2, t.b3, t.b4 );
   printf("c %10.4f %10.4f %10.4f %10.4f \n", t.c1, t.c2, t.c3, t.c4 );
   printf("d %10.4f %10.4f %10.4f %10.4f \n", t.d1, t.d2, t.d3, t.d4 );
}

void AssimpNode::setParent(AssimpNode* parent)
{
    m_parent = parent ;
}
void AssimpNode::setIndex(unsigned int index)
{
    //printf("AssimpNode::setIndex %d \n", index );
    m_index = index ; 
}
void AssimpNode::setDepth(unsigned int depth)
{
    printf("AssimpNode::setDepth %d \n", depth );
    m_depth = depth ; 
}



void AssimpNode::setTransform(aiMatrix4x4 transform)
{
    m_transform = transform ; 
    //dumpTransform(m_transform);
}

unsigned int AssimpNode::getIndex()
{
    return m_index ; 
}
unsigned int AssimpNode::getDepth()
{
    return m_depth ; 
}

void AssimpNode::addChild(AssimpNode* child)
{
    m_children.push_back(child); 
}

AssimpNode* AssimpNode::getParent()
{
    return m_parent ;  
}

unsigned int AssimpNode::getNumChildren()
{
    return m_children.size(); 
}

AssimpNode* AssimpNode::getChild(unsigned int n)
{
    if(n < getNumChildren()) return m_children[n] ; 
    return NULL ;  
}



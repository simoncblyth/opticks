#include "GNode.hh"
#include "GMesh.hh"

#include "stdio.h"
#include "stdlib.h"


GNode::GNode(unsigned int index, GMatrixF* transform, GMesh* mesh) :
    m_index(index), 
    m_transform(transform),
    m_mesh(mesh),
    m_parent(NULL),
    m_description(NULL),
    m_low(NULL),
    m_high(NULL)
{
    updateBounds();
}

GNode::~GNode()
{
    free(m_description);
}

unsigned int GNode::getIndex()
{
    return m_index ; 
}

void GNode::Summary(const char* msg)
{
    printf("%s idx %u nchild %u \n", msg, m_index, getNumChildren());
}

void GNode::setParent(GNode* parent)
{ 
    m_parent = parent ; 
}

void GNode::setDescription(char* description)
{ 
    m_description = strdup(description) ; 
}

char* GNode::getDescription()
{
    return m_description ;
}

void GNode::addChild(GNode* child)
{
    m_children.push_back(child);
}

GNode* GNode::getParent()
{
    return m_parent ; 
}

GNode* GNode::getChild(unsigned int index)
{
    return index < getNumChildren() ? m_children[index] : NULL ;
}

unsigned int GNode::getNumChildren()
{
    return m_children.size();
}

 

void GNode::updateBounds(gfloat3& low, gfloat3& high )
{
    m_mesh->updateBounds(low, high, *m_transform); 
}

void GNode::updateBounds()
{
    gfloat3  low( 1e10f, 1e10f, 1e10f);
    gfloat3 high( -1e10f, -1e10f, -1e10f);

    updateBounds(low, high);

    /*
    for( unsigned int i = 0; i < getNumChildren() ;++i )
    {
        GNode* child = getChild(i);
        child->updateBounds(low, high);
    }
    */

    m_low = new gfloat3(low.x, low.y, low.z) ;
    m_high = new gfloat3(high.x, high.y, high.z);
}

gfloat3* GNode::getLow()
{
    return m_low ; 
}
gfloat3* GNode::getHigh()
{
    return m_high ; 
}

GMesh* GNode::getMesh()
{
   return m_mesh ;
}
GMatrixF* GNode::getTransform()
{
   return m_transform ;
}





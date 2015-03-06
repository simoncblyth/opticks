#include "GNode.hh"
#include "stdio.h"
#include "stdlib.h"

GNode::GNode(unsigned int index) :
    m_index(index), 
    m_parent(NULL),
    m_description(NULL)
{
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

 



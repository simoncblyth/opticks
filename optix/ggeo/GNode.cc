#include "GNode.hh"
#include "GSolid.hh"
#include "stdio.h"


// is this needed ?  could use to defer flattening 

GNode::GNode() : 
    m_solid(NULL)
{
}

GNode::~GNode()
{
}

void GNode::Summary(const char* msg)
{
    printf("%s\n", msg);
}

unsigned int GNode::getNumChildren()
{
    return m_children.size();
}

GNode* GNode::getChild(unsigned int n)
{
    return n < getNumChildren() ? m_children[n] : NULL ;
}
  



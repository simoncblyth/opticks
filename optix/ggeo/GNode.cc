#include "GNode.hh"
#include "GMesh.hh"

#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include <algorithm>


GNode::GNode(unsigned int index, GMatrixF* transform, GMesh* mesh) :
    m_index(index), 
    m_transform(transform),
    m_mesh(mesh),
    m_parent(NULL),
    m_description(NULL),
    m_node_indices(NULL),
    m_substance_indices(NULL),
    m_low(NULL),
    m_high(NULL)
{
    updateBounds();
    setNodeIndices(m_index);
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





unsigned int* GNode::getSubstanceIndices()
{
    return m_substance_indices ; 
}
void GNode::setSubstanceIndices(unsigned int* substance_indices)
{
    m_substance_indices = substance_indices ; 
}


void GNode::setMeshSubstance(unsigned int index)
{
    // unsigned int* array of the substance index repeated nface times

    unsigned int nface = m_mesh->getNumFaces(); 
    unsigned int* indices = new unsigned int[nface] ;
    while(nface--) indices[nface] = index ; 
    setSubstanceIndices(indices);
}


unsigned int* GNode::getNodeIndices()
{
    return m_node_indices ; 
}

void GNode::setNodeIndices(unsigned int index)
{
    // unsigned int* array of the node index repeated nface times

    unsigned int nface = m_mesh->getNumFaces(); 
    unsigned int* indices = new unsigned int[nface] ;
    while(nface--) indices[nface] = index ; 

    //m_mesh->setNodes(indices);
    m_node_indices = indices ; 
}

// duplication done in the above setters seems silly, 
// but this is to allow simple merging when flatten a tree 
// of nodes into a single structure


void GNode::updateDistinctSubstanceIndices()
{
    for(unsigned int i=0 ; i < m_mesh->getNumFaces() ; i++)
    {
        unsigned int index = m_substance_indices[i] ;
        if(std::count(m_distinct_substance_indices.begin(), m_distinct_substance_indices.end(), index ) == 0) m_distinct_substance_indices.push_back(index);
    }  
    std::sort( m_distinct_substance_indices.begin(), m_distinct_substance_indices.end() );
}
 
std::vector<unsigned int>& GNode::getDistinctSubstanceIndices()
{
    if(m_distinct_substance_indices.size()==0) updateDistinctSubstanceIndices();
    return m_distinct_substance_indices ;
}


#include "GNode.hh"
#include "GMesh.hh"

#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include <algorithm>
#include <iostream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GNode::init()
{
    updateBounds();
    setNodeIndices(m_index);
}

GNode::~GNode()
{
    free(m_description);
}

void GNode::Summary(const char* msg)
{
    printf("%s idx %u nchild %u \n", msg, m_index, getNumChildren());
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

    m_low = new gfloat3(low.x, low.y, low.z) ;
    m_high = new gfloat3(high.x, high.y, high.z);
}


void GNode::setBoundaryIndices(unsigned int index)
{
    // unsigned int* array of the boundary index repeated nface times
    unsigned int nface = m_mesh->getNumFaces(); 
    unsigned int* indices = new unsigned int[nface] ;
    while(nface--) indices[nface] = index ; 
    setBoundaryIndices(indices);
}
void GNode::setNodeIndices(unsigned int index)
{
    // unsigned int* array of the node index repeated nface times
    unsigned int nface = m_mesh->getNumFaces(); 
    unsigned int* indices = new unsigned int[nface] ;
    while(nface--) indices[nface] = index ; 
    m_node_indices = indices ; 
}
void GNode::setSensorIndices(unsigned int index)
{
    // unsigned int* array of the node index repeated nface times
    unsigned int nface = m_mesh->getNumFaces(); 
    unsigned int* indices = new unsigned int[nface] ;
    while(nface--) indices[nface] = index ; 
    m_sensor_indices = indices ; 
}

void GNode::updateDistinctBoundaryIndices()
{
    for(unsigned int i=0 ; i < m_mesh->getNumFaces() ; i++)
    {
        unsigned int index = m_boundary_indices[i] ;
        if(std::count(m_distinct_boundary_indices.begin(), m_distinct_boundary_indices.end(), index ) == 0) m_distinct_boundary_indices.push_back(index);
    }  
    std::sort( m_distinct_boundary_indices.begin(), m_distinct_boundary_indices.end() );
}
 
std::vector<unsigned int>& GNode::getDistinctBoundaryIndices()
{
    if(m_distinct_boundary_indices.size()==0) updateDistinctBoundaryIndices();
    return m_distinct_boundary_indices ;
}




std::vector<GNode*> GNode::getAncestors(bool reverse)
{
    std::vector<GNode*> ancestors ;  
    GNode* node = getParent();
    while(node)
    {
        ancestors.push_back(node);
        node = node->getParent();
    }
    if(reverse) std::reverse( ancestors.begin(), ancestors.end() );
    return ancestors ; 
}

GMatrixF* GNode::calculateTransform()
{
    bool reverse = true ; 
    std::vector<GNode*> nodes = getAncestors(reverse);
    nodes.push_back(this);

    typedef std::vector<GNode*>::const_iterator NIT ; 
    //LOG(info) << "GNode::calculateTransform " ; 

    GMatrix<float>* m = new GMatrix<float> ;
    //unsigned int idx(0);
    for(NIT it=nodes.begin() ; it != nodes.end() ; it++)
    {
        GNode* node = *it ; 
        //std::cout << std::setw(3) << idx << node->getName() << std::endl ; 
        (*m) *= (*node->getLevelTransform()); 
    }
    return m ; 
}

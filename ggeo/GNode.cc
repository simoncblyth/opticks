/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>

// sysrap-
#include "SDigest.hh"

#include "NGLM.hpp"

// ggeo-
#include "GMatrix.hh"
#include "GMesh.hh"
#include "GNode.hh"
#include "GVolume.hh"


#include "PLOG.hh"
// trace/debug/info/warning/error/fatal



GNode::GNode(unsigned int index, GMatrixF* transform, const GMesh* mesh) 
    :
    m_selfdigest(true),
    m_selected(true),
    m_index(index), 
    m_parent(NULL),
    m_description(NULL),
    m_transform(transform),
    m_ltransform(NULL),
    m_gtriple(NULL),
    m_ltriple(NULL),
    m_mesh(mesh),
    m_low(NULL),
    m_high(NULL),
    m_boundary_indices(NULL),
    m_sensor_indices(NULL),
    m_node_indices(NULL),
    m_name(NULL),
    m_progeny_count(0),
    m_remainder_progeny_count(0),
    m_repeat_index(0),
    m_progeny_num_vertices(0)
{
    init();
}


void GNode::setIndex(unsigned int index)
{
    m_index = index ; 
}

void GNode::setSelected(bool selected)
{
    m_selected = selected ; 
}
bool GNode::isSelected() const 
{
   return m_selected ; 
}



gfloat3* GNode::getLow()
{
    return m_low ; 
}
gfloat3* GNode::getHigh()
{
    return m_high ; 
}
const GMesh* GNode::getMesh() const 
{
   return m_mesh ;
}
unsigned GNode::getMeshIndex() const 
{
    assert(m_mesh);
    return m_mesh ? m_mesh->getIndex() : 0 ;
}


glm::mat4 GNode::getTransformMat4() const 
{
    float* f = (float*)m_transform->getPointer();
    assert(f);
    return glm::make_mat4(f);  
}





GMatrixF* GNode::getTransform() const 
{
   return m_transform ;
}

unsigned int* GNode::getBoundaryIndices() const 
{
    return m_boundary_indices ; 
}
unsigned int* GNode::getNodeIndices() const 
{
    return m_node_indices ; 
}
unsigned int* GNode::getSensorIndices() const 
{
    return m_sensor_indices ; 
}



void GNode::setBoundaryIndices(unsigned int* boundary_indices)
{
    m_boundary_indices = boundary_indices ; 
}
unsigned int GNode::getIndex() const 
{
    return m_index ; 
}
void GNode::setParent(GNode* parent)
{ 
    m_parent = parent ; 
}
GNode* GNode::getParent() const 
{
    return m_parent ; 
}
char* GNode::getDescription() const 
{
    return m_description ;
}
void GNode::setDescription(char* description)
{ 
    m_description = strdup(description) ; 
}
void GNode::addChild(GNode* child)
{
    m_children.push_back(child);
}
GNode* GNode::getChild(unsigned index) const 
{
    return index < getNumChildren() ? m_children[index] : NULL ;
}

GVolume* GNode::getChildVolume(unsigned index) const 
{
    return dynamic_cast<GVolume*>(getChild(index));
}



unsigned GNode::getNumChildren() const 
{
    return m_children.size();
}

void GNode::setLevelTransform(GMatrixF* ltransform)
{
   m_ltransform = ltransform ; 
}
GMatrixF* GNode::getLevelTransform() const 
{
   return m_ltransform ; 
}




void GNode::setLocalTransform(const nmat4triple* ltriple)
{
    m_ltriple = ltriple ; 
}
void GNode::setGlobalTransform(const nmat4triple* gtriple)
{
    m_gtriple = gtriple ; 
}
const nmat4triple* GNode::getLocalTransform() const 
{
    return m_ltriple ; 
}
const nmat4triple* GNode::getGlobalTransform() const 
{
    return m_gtriple ; 
}
 







void GNode::setName(const char* name)
{
    m_name = strdup(name); 
}
const char* GNode::getName() const 
{
    return m_name ; 
}
void GNode::setRepeatIndex(unsigned int index)
{
    m_repeat_index = index ; 
}
unsigned int GNode::getRepeatIndex() const 
{
    return m_repeat_index ; 
}











void GNode::init()
{
    if(!m_mesh)
    {
        LOG(error) << "GNode::init mesh NULL " ; 
        return ; 
    } 

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

void GNode::dump(const char* )
{
    //LOG(info) << msg ; 
    //printf("%s idx %u nchild %u \n", msg, m_index, getNumChildren());

    if(m_mesh)
    {
         unsigned msol = m_mesh->getNumVolumes() ;
         gfloat4 mce0 = m_mesh->getCenterExtent(0);
         LOG(info) << "mesh.numVolumes " << msol << " mesh.ce.0 " << mce0.description() ; 

         for(unsigned i=0 ; i < msol ; i++)
         {
             gfloat4 mce = m_mesh->getCenterExtent(i);
             std::cout << std::setw(4) << i 
                       << " mesh.ce " << mce.description()
                       << std::endl ; 
         }
    }   
}
 
 

void GNode::updateBounds(gfloat3& low, gfloat3& high )
{
   // TODO: reorg to avoid this... 
    const_cast<GMesh*>(m_mesh)->updateBounds(low, high, *m_transform); 
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

std::vector<GNode*>& GNode::getAncestors()
{
    if(m_ancestors.size() == 0 )
    { 
        GNode* node = getParent();
        while(node)
        {
            m_ancestors.push_back(node);
            node = node->getParent();
        }
        std::reverse( m_ancestors.begin(), m_ancestors.end() );
    }
    return m_ancestors ; 
}


/**
GNode::getProgeny
------------------

Returns the progeny of a node.
Collects into m_progeny which avoids repeating 
the collection. 

Collection starts from the children, 
as wish to avoid collecting the start node.  

**/

std::vector<GNode*>& GNode::getProgeny()
{
    if(m_progeny.size() == 0)
    {
        for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(m_progeny); 
        m_progeny_count = m_progeny.size();
    }
    return m_progeny ; 
}

void GNode::collectProgeny(std::vector<GNode*>& progeny) 
{
    progeny.push_back(this);
    for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(progeny);
}


/**
GNode::getRemainderProgeny
-------------------------

Returns the remainder progeny of a node, ie with repeat index zero.
Collects into m_remainder_progeny which avoids repeating 
the collection. 

Collection starts from the children, as wish to avoid collecting the start node.  

**/

std::vector<GNode*>& GNode::getRemainderProgeny()
{
    if(m_remainder_progeny.size() == 0)
    {
        for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectRemainderProgeny(m_remainder_progeny); 
        m_remainder_progeny_count = m_remainder_progeny.size();
    }
    return m_remainder_progeny ; 
}

void GNode::collectRemainderProgeny(std::vector<GNode*>& remainder_progeny) 
{
    if(getRepeatIndex() == 0)
    {
        remainder_progeny.push_back(this);
    }
    for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectRemainderProgeny(remainder_progeny);
}



/**
GNode::getRelativeTransform
-----------------------------

This effectively provides transforms for nodes within 
a subtree as if the base of the subtree were the root node.

A vector of nodes is constructed starting from the root 
node and ending with *this* node.  This node must be 
within the subtree starting from the base node otherwise
this will assert.

The level transforms from nodes within the base node 
(but not the base node transform itself) are mutiplied together to 
form the relative to base node transform.

This is canonically used by GMergedMesh::mergeVolume for 
concatenating multiple GVolume for repeated instance where the 
base node is the outermost node of an example of the repeated 
instance. 

Only transforms after the base node are collected
   
#. When this node is the base get identity ?
#. When this node is above base this will assert
#. When this node is below base get transforms relative to base
   
**/


GMatrixF* GNode::getRelativeTransform(const GNode* base)  // cannot be const due to getAncestors caching 
{
    std::vector<GNode*> nodes = getAncestors();  // <--- in order starting from root
    nodes.push_back(this);

    typedef std::vector<GNode*>::const_iterator NIT ; 
    //LOG(info) << "GNode::calculateTransform " ; 

    GMatrix<float>* m = new GMatrix<float> ;

    unsigned int nbase(0);
    unsigned int nprod(0);
    bool collect = false ; 

    for(NIT it=nodes.begin() ; it != nodes.end() ; it++)
    {
        GNode* node = *it ; 
        if(collect)
        {
            //std::cout << std::setw(3) << idx << node->getName() << std::endl ; 
            (*m) *= (*node->getLevelTransform()); 
            nprod++ ; 
        }
        if(node == base)
        {
           nbase++ ; 
           collect = true ; 
        }        
    }

    if(nbase == 0)
    {
        LOG(fatal)
            << " BASE NODE IS NOT ANCESTOR " 
            << " base node index: " << ( base ? base->getIndex() : -1 )
            << " base node name:  " << base->getName()
            << " this node index:  " << getIndex() 
            << " this node name:  " << this->getName()
            ;
        assert(0);
    }
    return m ; 
}




/**
GNode::getRelativeVerticesBBox
---------------------------------

For base NULL the bbox is in global coordinate system, 
otherwise the coordinate system of the given base node is used.
See getRelativeTransform.

**/

nbbox* GNode::getRelativeVerticesBBox( const GNode* base ) // cannot be const due to getRelativeTransform/getAncestors
{
    const GMesh* mesh = getMesh();  
    unsigned num_vert = mesh->getNumVertices();
    GMatrixF* transform = getRelativeTransform(base) ; 
    gfloat3* vertices = mesh->getTransformedVertices(*transform) ;
    nbbox* bb = GMesh::findBBox_( vertices, num_vert ); 
    return bb ; 
}
nbbox* GNode::getVerticesBBox() const 
{
    const GMesh* mesh = getMesh();  
    unsigned num_vert = mesh->getNumVertices();
    GMatrixF* transform = getTransform() ; 
    gfloat3* vertices = mesh->getTransformedVertices(*transform) ;
    nbbox* bb = GMesh::findBBox_( vertices, num_vert ); 
    return bb ; 
}




GMatrixF* GNode::calculateTransform()
{
    std::vector<GNode*> nodes = getAncestors();
    nodes.push_back(this);    // ancestors + self

    typedef std::vector<GNode*>::const_iterator NIT ; 

    GMatrix<float>* m = new GMatrix<float> ;
    for(NIT it=nodes.begin() ; it != nodes.end() ; it++)
    {
        GNode* node = *it ; 
        //std::cout << std::setw(3) << idx << node->getName() << std::endl ; 
        (*m) *= (*node->getLevelTransform()); 
    }
    return m ; 
}

std::string GNode::localDigest()  
{
    GMatrix<float>* t = getLevelTransform();
    std::string tdig = t->digest();

    char meshidx[8];
    snprintf(meshidx, 8, "%u", m_mesh->getIndex());

    SDigest dig ;
    dig.update( (char*)tdig.c_str(), strlen(tdig.c_str()) ); 
    dig.update( meshidx , strlen(meshidx) ); 
    return dig.finalize();
}


std::string GNode::meshDigest()  
{
    char meshidx[8];
    snprintf(meshidx, 8, "%u", m_mesh->getIndex());
    return meshidx ; 
}


std::string GNode::localDigest(std::vector<GNode*>& nodes, GNode* extra)
{
    SDigest dig ;
    for(unsigned int i=0 ; i < nodes.size() ; i++)
    {
        GNode* node = nodes[i];
        std::string nd = node->localDigest();
        dig.update( (char*)nd.c_str(), strlen(nd.c_str()) ); 
    } 

    if(extra)
    {
        //std::string xd = extra->localDigest();   incorporates levelTransform and meshIndex
        std::string xd = extra->meshDigest();
        dig.update( (char*)xd.c_str(), strlen(xd.c_str()) ); 
    }

    return dig.finalize();
}


std::string& GNode::getLocalDigest()
{
    if(m_local_digest.empty())
    {
         m_local_digest = localDigest();
    }
    return m_local_digest ; 
}


unsigned int GNode::getProgenyNumVertices()
{
    if(m_progeny_num_vertices == 0)
    {
        std::vector<GNode*>& progeny = getProgeny();
        typedef std::vector<GNode*>::const_iterator NIT ; 
        unsigned int num_vertices(0);
        for(NIT it=progeny.begin() ; it != progeny.end() ; it++)
        {
            GNode* node = *it ; 
            const GMesh* mesh = node->getMesh();
            num_vertices += mesh->getNumVertices();
        }
        GNode* extra = m_selfdigest ? this : NULL ; 
        if(extra)
        {
            num_vertices += m_mesh->getNumVertices(); 
        }
        m_progeny_num_vertices = num_vertices ; 
    }
    return m_progeny_num_vertices ; 
}


std::string& GNode::getProgenyDigest()
{
    if(m_progeny_digest.empty())
    {
        std::vector<GNode*>& progeny = getProgeny();
        GNode* extra = m_selfdigest ? this : NULL ; 
        m_progeny_digest = GNode::localDigest(progeny, extra) ; 
    }
    return m_progeny_digest ;
}


/**
GNode::getPriorProgenyCount
-------------------------------

Former name of getLastProgenyCount too similar to other methods of different meaning.

* CAUTION : only set at the first getProgeny call on a node

**/

unsigned GNode::getPriorProgenyCount() const 
{
    return m_progeny_count ; 
}

/**
GNode::getPriorGlobalProgenyCount
----------------------------------

* CAUTION : only set at the first getGlobalProgeny call on a node

**/

unsigned GNode::getPriorRemainderProgenyCount() const 
{
    return m_remainder_progeny_count ; 
}




/**
GNode::findProgenyDigest
-------------------------

If digest of this node matches target return this node
otherwise recursively invoke on children returning the first match.

Hmm slightly funny structure, preorder ?

**/

GNode* GNode::findProgenyDigest(const std::string& dig)  
{
   std::string& pdig = getProgenyDigest();
   GNode* node = NULL ; 
   if(strcmp(pdig.c_str(), dig.c_str())==0)
   {
       node = this ;
   }
   else
   {
       for(unsigned i = 0; i < getNumChildren(); i++) 
       {
           GNode* child = getChild(i);
           node = child->findProgenyDigest(dig);
           if(node) break ;
       }
   }
   return node ; 
}


std::vector<const GNode*> GNode::findAllProgenyDigest(std::string& dig)
{
    std::vector<const GNode*> match ;
    collectAllProgenyDigest(match, dig );
    return match ;
}
std::vector<const GNode*> GNode::findAllInstances(unsigned ridx, bool inside_self, bool honour_selection)
{
    std::vector<const GNode*> match ;
    collectAllInstances(match, ridx, inside_self, honour_selection );
    return match ;
}


void GNode::collectAllProgenyDigest(std::vector<const GNode*>& match, std::string& dig)
{
    std::string& pdig = getProgenyDigest();
    if(strcmp(pdig.c_str(), dig.c_str())==0) 
    {
        match.push_back(this);
    }
    else
    {
        for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectAllProgenyDigest(match, dig );
    }
}


/**
GNode::collectAllInstances
---------------------------

* with "inside=false" the recursion stops at the first matched instance,.
* with "inside=true" the recursion continues, collecting all matched instances

**/

void GNode::collectAllInstances(std::vector<const GNode*>& match, unsigned ridx, bool inside, bool honour_selection )
{
    bool matched_ridx = getRepeatIndex()==ridx ; 
    bool matched_selection = honour_selection ? m_selected : true ; 
    bool matched = matched_ridx && matched_selection ;

    if(inside)
    {
        if(matched) match.push_back(this);
        for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectAllInstances(match, ridx, inside, honour_selection );
    }
    else
    {
        if(matched) 
        {
            match.push_back(this);
        }
        else
        {
            for(unsigned i = 0; i < getNumChildren(); i++) getChild(i)->collectAllInstances(match, ridx, inside, honour_selection );
        }
    }
}




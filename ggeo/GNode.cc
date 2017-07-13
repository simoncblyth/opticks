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
#include "GSolid.hh"


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
    m_mesh(mesh),
    m_low(NULL),
    m_high(NULL),
    m_boundary_indices(NULL),
    m_sensor_indices(NULL),
    m_node_indices(NULL),
    m_name(NULL),
    m_progeny_count(0),
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
bool GNode::isSelected()
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
const GMesh* GNode::getMesh() 
{
   return m_mesh ;
}
unsigned GNode::getMeshIndex() const 
{
    assert(m_mesh);
    return m_mesh ? m_mesh->getIndex() : 0 ;
}


glm::mat4 GNode::getTransformMat4()
{
    float* f = (float*)m_transform->getPointer();
    assert(f);
    return glm::make_mat4(f);  
}


GMatrixF* GNode::getTransform()
{
   return m_transform ;
}

unsigned int* GNode::getBoundaryIndices()
{
    return m_boundary_indices ; 
}
unsigned int* GNode::getNodeIndices()
{
    return m_node_indices ; 
}
unsigned int* GNode::getSensorIndices()
{
    return m_sensor_indices ; 
}



void GNode::setBoundaryIndices(unsigned int* boundary_indices)
{
    m_boundary_indices = boundary_indices ; 
}
unsigned int GNode::getIndex()
{
    return m_index ; 
}
void GNode::setParent(GNode* parent)
{ 
    m_parent = parent ; 
}
GNode* GNode::getParent()
{
    return m_parent ; 
}
char* GNode::getDescription()
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
GNode* GNode::getChild(unsigned int index)
{
    return index < getNumChildren() ? m_children[index] : NULL ;
}

GSolid* GNode::getChildSolid(unsigned int index)
{
    return dynamic_cast<GSolid*>(getChild(index));
}



unsigned int GNode::getNumChildren()
{
    return m_children.size();
}

void GNode::setLevelTransform(GMatrixF* ltransform)
{
   m_ltransform = ltransform ; 
}
GMatrixF* GNode::getLevelTransform()
{
   return m_ltransform ; 
}

void GNode::setName(const char* name)
{
    m_name = strdup(name); 
}
const char* GNode::getName()
{
    return m_name ; 
}
void GNode::setRepeatIndex(unsigned int index)
{
    m_repeat_index = index ; 
}
unsigned int GNode::getRepeatIndex()
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
         unsigned msol = m_mesh->getNumSolids() ;
         gfloat4 mce0 = m_mesh->getCenterExtent(0);
         LOG(info) << "mesh.numSolids " << msol << " mesh.ce.0 " << mce0.description() ; 

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

std::vector<GNode*>& GNode::getProgeny()
{
    if(m_progeny.size() == 0)
    {
        // call on children, as wish to avoid collecting self  
        for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(m_progeny); 
    }
    return m_progeny ; 
}

void GNode::collectProgeny(std::vector<GNode*>& progeny)
{
    progeny.push_back(this);
    for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(progeny);
}


GMatrixF* GNode::getRelativeTransform(GNode* base)
{
   // only transforms after the base node are collected
   //
   // #. When this node is the base get identity ?
   // #. When this node is above base this will assert
   // #. When this node is below base get transforms relative to base
   //  

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
        m_progeny_count = progeny.size();
        GNode* extra = m_selfdigest ? this : NULL ; 
        m_progeny_digest = GNode::localDigest(progeny, extra) ; 
    }
    return m_progeny_digest ;
}

unsigned int GNode::getProgenyCount()
{
    return m_progeny_count ; 
}

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
       for(unsigned int i = 0; i < getNumChildren(); i++) 
       {
           GNode* child = getChild(i);
           node = child->findProgenyDigest(dig);
           if(node) break ;
       }
   }
   return node ; 
}


std::vector<GNode*> GNode::findAllProgenyDigest(std::string& dig)
{
    std::vector<GNode*> match ;
    collectAllProgenyDigest(match, dig );
    return match ;
}
std::vector<GNode*> GNode::findAllInstances(unsigned ridx, bool inside_self, bool honour_selection)
{
    std::vector<GNode*> match ;
    collectAllInstances(match, ridx, inside_self, honour_selection );
    return match ;
}




void GNode::collectAllProgenyDigest(std::vector<GNode*>& match, std::string& dig)
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

void GNode::collectAllInstances(std::vector<GNode*>& match, unsigned ridx, bool inside, bool honour_selection )
{
    // NB when a node labelled with the target ridx is found... 
    // the traverse stops there...  
    // ie there is no attempt to collect from inside the target node
    // ... this is why looking for instances of ridx 0 returns just one instance, 
    // the root node
    //

     
    bool matched_ridx = getRepeatIndex()==ridx ; 
    bool matched_selection = honour_selection ? m_selected : true ; 
    bool matched = matched_ridx && matched_selection ;

    if(inside)
    {
        if(matched) match.push_back(this);
        for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectAllInstances(match, ridx, inside, honour_selection );
    }
    else
    {
        // without "inside" the traverse is stopped at the first matched instance, ie it doesnt look inside self 
        if(matched) 
        {
            match.push_back(this);
        }
        else
        {
            for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectAllInstances(match, ridx, inside, honour_selection );
        }
    }
}

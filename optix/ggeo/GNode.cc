#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>

// brap-
#include "BDigest.hh"
#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

// ggeo-
#include "GNode.hh"
#include "GMesh.hh"


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
    std::vector<GNode*> nodes = getAncestors();
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
        LOG(warning)<<"GNode::getRelativeTransform " 
                    << " base node  " << base->getName()
                    << " is not an ancestor of this node  " << this->getName()
                    ;
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

    BDigest dig ;
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
    BDigest dig ;
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
            GMesh* mesh = node->getMesh();
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



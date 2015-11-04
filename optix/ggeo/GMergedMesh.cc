#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"

// npy-
#include "Timer.hpp"
#include "NSensor.hpp"

#include <climits>
#include <iostream>
#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo, GNode* base)
{

    Timer t("GMergedMesh::create") ; 
    t.setVerbose(false);
    t.start();

    GMergedMesh* mm = new GMergedMesh( index ); 

    if(base == NULL)  // non-instanced global transforms
    {
        mm->setCurrentBase(NULL);
        base = static_cast<GNode*>(ggeo->getSolid(0)); 
        unsigned int numMeshes = ggeo->getNumMeshes();
        assert(numMeshes < 500 );
    }
    else   // instances transforms, with transform heirarchy split at the base 
    {
        mm->setCurrentBase(base);
    }

    LOG(info)<<"GMergedMesh::create"
             << " index " << index 
             << " from base " << base->getName() ;
             ; 

    // 1st pass traversal : counts vertices and faces

    mm->traverse( base, 0, PASS_COUNT );  

    t("1st pass traverse");

    // allocate space for flattened arrays

    LOG(info) << "GMergedMesh::create" 
              << " index " << index 
              << " numVertices " << mm->getNumVertices()
              << " numFaces " << mm->getNumFaces()
              << " numSolids " << mm->getNumSolids()
              << " numSolidsSelected " << mm->getNumSolidsSelected()
              ;

    mm->allocate(); 

    // 2nd pass traversal : merge copy GMesh into GMergedMesh 

    mm->traverse( base, 0, PASS_MERGE );  
    t("2nd pass traverse");

    mm->updateBounds();

    t("updateBounds");

    t.stop();
    //t.dump();

    return mm ;
}


void GMergedMesh::count( GSolid* solid, bool selected )
{
    if(!selected) return ; 

    GMesh* mesh = solid->getMesh();
    unsigned int nface = mesh->getNumFaces();
    unsigned int nvert = mesh->getNumVertices();
    unsigned int meshIndex = mesh->getIndex();

    m_num_vertices += nvert ;
    m_num_faces    += nface ; 
    m_mesh_usage[meshIndex] += 1 ;  // which meshes contribute to the mergedmesh
}

void GMergedMesh::merge( GSolid* solid, bool selected )
{
    GMesh* mesh = solid->getMesh();
    unsigned int nvert = mesh->getNumVertices();
    unsigned int nface = mesh->getNumFaces();

    GNode* base = getCurrentBase();
    GMatrixF* transform = base ? solid->getRelativeTransform(base) : solid->getTransform() ;    
    gfloat3* vertices = mesh->getTransformedVertices(*transform) ;

    unsigned int boundary = solid->getBoundary();
    NSensor* sensor = solid->getSensor();

    unsigned int nodeIndex = solid->getIndex();
    unsigned int meshIndex = mesh->getIndex();
    unsigned int sensorIndex = NSensor::RefIndex(sensor) ; 
    guint4 _identity = solid->getIdentity();
    assert(_identity.x == nodeIndex);
    assert(_identity.y == meshIndex);
    assert(_identity.z == boundary);
    //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
    
    LOG(info) << "GMergedMesh::merge"
              << " nodeIndex " << nodeIndex
              << " boundaryIndex " << boundary
              << " sensorIndex " << sensorIndex
              << " sensor " << ( sensor ? sensor->description() : "NULL" )
              ;

    GNode* parent = solid->getParent();
    unsigned int parentIndex = parent ? parent->getIndex() : UINT_MAX ;

    // needs to be outside the selection branch for the all solid center extent
    gbbox bb = GMesh::findBBox(vertices, nvert) ;

    m_bbox[m_cur_solid] = bb ;  
    m_center_extent[m_cur_solid] = bb.center_extent() ;
    transform->copyTo( getTransform(m_cur_solid) );

    m_meshes[m_cur_solid] = meshIndex ; 

    // face and vertex counts must use same selection as above to be usable 
    // with the above filled vertices and indices 

    m_nodeinfo[m_cur_solid].x = selected ? nface : 0 ; 
    m_nodeinfo[m_cur_solid].y = selected ? nvert : 0 ; 
    m_nodeinfo[m_cur_solid].z = nodeIndex ;  
    m_nodeinfo[m_cur_solid].w = parentIndex ; 

    if(isGlobal())
         assert(nodeIndex == m_cur_solid);

    m_identity[m_cur_solid] = _identity ; 

    if(selected)
    {
        gfloat3* normals = mesh->getTransformedNormals(*transform);  
        guint3* faces = mesh->getFaces();

        for(unsigned int i=0 ; i<nvert ; ++i )
        {
            m_vertices[m_cur_vertices+i] = vertices[i] ; 
            m_normals[m_cur_vertices+i] = normals[i] ; 
        }

        // TODO: consolidate into uint4 (with one spare)
        unsigned int* node_indices = solid->getNodeIndices();
        unsigned int* boundary_indices = solid->getBoundaryIndices();
        unsigned int* sensor_indices = solid->getSensorIndices();
        assert(node_indices);
        assert(boundary_indices);
        assert(sensor_indices);

        // offset the vertex indices as are combining all meshes into single vertex list 
        for(unsigned int i=0 ; i<nface ; ++i )
        {
            m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
            m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
            m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

            m_nodes[m_cur_faces+i]      = node_indices[i] ;
            m_boundaries[m_cur_faces+i] = boundary_indices[i] ;
            m_sensors[m_cur_faces+i]    = sensor_indices[i] ;
        }

        // offset within the flat arrays
        m_cur_vertices += nvert ;
        m_cur_faces    += nface ;
    }
}


void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GNode* base = getCurrentBase();
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    // using repeat index labelling in the tree
    bool repsel = getIndex() == -1 || solid->getRepeatIndex() == getIndex() ;
    bool selected = solid->isSelected() && repsel ;

    if(pass == PASS_COUNT )
    {
        count(solid, selected);
        m_num_solids += 1 ; 
        if(selected) m_num_solids_selected += 1;
    }
    else if(pass == PASS_MERGE )
    {
        merge(solid, selected);
        m_cur_solid += 1 ;    // irrespective of selection, as prefer absolute solid indexing 
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}


GMergedMesh* GMergedMesh::combine(unsigned int index, GMergedMesh* mm, std::vector<GSolid*>& solids)
{
    return NULL ; 
}


void GMergedMesh::reportMeshUsage(GGeo* ggeo, const char* msg)
{
     LOG(info) << msg ; 
     typedef std::map<unsigned int, unsigned int>::const_iterator MUUI ; 

     unsigned int tv(0) ; 
     for(MUUI it=m_mesh_usage.begin() ; it != m_mesh_usage.end() ; it++)
     {
         unsigned int meshIndex = it->first ; 
         unsigned int nodeCount = it->second ; 
 
         GMesh* mesh = ggeo->getMesh(meshIndex);
         const char* meshName = mesh->getName() ; 
         unsigned int nv = mesh->getNumVertices() ; 
         unsigned int nf = mesh->getNumFaces() ; 

         printf("  %4d (v%5d f%5d) : %6d : %7d : %s \n", meshIndex, nv, nf, nodeCount, nodeCount*nv, meshName);

         tv += nodeCount*nv ; 
     }
     printf(" tv : %7d \n", tv);
}


GMergedMesh* GMergedMesh::load(const char* dir, unsigned int index, const char* version)
{
    GMergedMesh* mm(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        LOG(warning) << "GMergedMesh::load directory DOES NOT EXIST " <<  dir ;
    }
    else
    {
        mm = new GMergedMesh(index);
        if(index == 0) mm->setVersion(version);  // mesh versioning applies to  global buffer 
        mm->loadBuffers(dir);
    }
    return mm ; 
}


void GMergedMesh::dumpSolids(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int index=0 ; index < getNumSolids() ; ++index)
    {
        if(index % 1000 != 0) continue ; 
        gfloat4 ce = getCenterExtent(index) ;
        printf("  %u :    center %10.3f %10.3f %10.3f   extent %10.3f \n", index, ce.x, ce.y, ce.z, ce.w ); 
    }
}

float* GMergedMesh::getModelToWorldPtr(unsigned int index)
{
    return index == 0 ? GMesh::getModelToWorldPtr(0) : NULL ;
}



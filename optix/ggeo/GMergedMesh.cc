#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GBoundaryLib.hh"
#include "GBoundary.hh"
#include "GSensor.hh"

#include "Timer.hpp"

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



void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GNode* base = getCurrentBase();

    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GBoundary* boundary = solid->getBoundary();
    GSensor* sensor = solid->getSensor();
    GMesh* mesh = solid->getMesh();

    unsigned int nodeIndex = node->getIndex();
    unsigned int meshIndex = mesh->getIndex();
    unsigned int boundaryIndex = boundary->getIndex();
    unsigned int sensorIndex = GSensor::RefIndex(sensor) ; 

    guint4 _identity = solid->getIdentity();

    assert(_identity.x == nodeIndex);
    assert(_identity.y == meshIndex);
    assert(_identity.z == boundaryIndex);
    //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
    
    LOG(debug) << "GMergedMesh::traverse"
              << " nodeIndex " << nodeIndex
              << " sensorIndex " << sensorIndex
              << " sensor " << ( sensor ? sensor->description() : "NULL" )
              ;

    unsigned int nface = mesh->getNumFaces();
    unsigned int nvert = mesh->getNumVertices();

    GNode* parent = node->getParent();
    unsigned int parentIndex = parent ? parent->getIndex() : UINT_MAX ;

    // using repeat index labelling in the tree
    bool repsel = getIndex() == -1 || solid->getRepeatIndex() == getIndex() ;
    bool selected = solid->isSelected() && repsel ;

    // needs to be out here for the all solid center extent
    GMatrixF* transform = base ? node->getRelativeTransform(base) : node->getTransform() ;    
    gfloat3* vertices = mesh->getTransformedVertices(*transform) ;


    if(selected)
    {
        if(pass == PASS_COUNT )
        {
            m_num_vertices += nvert ;
            m_num_faces    += nface ; 
            m_mesh_usage[meshIndex] += 1 ;  // which meshes contribute to the mergedmesh
        }
        else if(pass == PASS_MERGE )
        {
            for(unsigned int i=0 ; i<nvert ; ++i )
            {
                m_vertices[m_cur_vertices+i] = vertices[i] ; 
            }

            // TODO: change transform to be the transpose of the inverse  ?
            // But so long as othonormal?, not needed.  
            // Ordinary rotation, translation, and uniform scaling are OK.
            //
            gfloat3* normals = mesh->getTransformedNormals(*transform);  
            for(unsigned int i=0 ; i<nvert ; ++i )
            {
                m_normals[m_cur_vertices+i] = normals[i] ; 
            }

            // offset the vertex indices as are combining all meshes into single vertex list 
            guint3* faces = mesh->getFaces();

            // NB from the GNode not the GMesh 
            // (there are only ~250 GMesh instances which are recycled by the ~12k GNode)

            // TODO: consolidate into uint4 (with one spare)
            unsigned int* node_indices = node->getNodeIndices();
            unsigned int* boundary_indices = node->getBoundaryIndices();
            unsigned int* sensor_indices = node->getSensorIndices();

            assert(node_indices);
            assert(boundary_indices);
            assert(sensor_indices);

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

        } // count or merge passes
    }     // selected



    // for all (not just selected) as prefer absolute solid indexing 
    if(pass == PASS_COUNT )
    {
        m_num_solids += 1 ; 
        if(selected) m_num_solids_selected += 1;
    }
    else if( pass == PASS_MERGE ) 
    {
        gbbox bb = GMesh::findBBox(vertices, nvert) ;

        m_bbox[m_cur_solid] = bb ;  
        m_center_extent[m_cur_solid] = bb.center_extent() ;

        transform->copyTo( getTransform(m_cur_solid) );

        m_meshes[m_cur_solid] = meshIndex ; 

        // face and vertex counts must use same selection as above to be usable 
        // with the above filled vertices and indices 

        m_nodeinfo[m_cur_solid].x = selected ? nface : 0 ; 
        m_nodeinfo[m_cur_solid].y = selected ? nvert : 0 ; 
        m_nodeinfo[m_cur_solid].z = nodeIndex ;  // redundant?
        m_nodeinfo[m_cur_solid].w = parentIndex ; 

        if(isGlobal())
        {
            assert(nodeIndex == m_cur_solid);
        }


         /*
        m_identity[m_cur_solid].x = nodeIndex ; 
        m_identity[m_cur_solid].y = meshIndex ; 
        m_identity[m_cur_solid].z = boundaryIndex ; 
        m_identity[m_cur_solid].w = sensorIndex ; 
        */
 
        m_identity[m_cur_solid] = _identity ; 

        m_cur_solid += 1 ; 
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}



void GMergedMesh::reportMeshUsage(GGeo* ggeo, const char* msg)
{
     printf("%s\n", msg);
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
        printf("GMergedMesh::load directory %s DOES NOT EXIST \n", dir);
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
    printf("%s\n", msg);
    for(unsigned int index=0 ; index < getNumSolids() ; ++index)
    {
        if(index % 1000 != 0) continue ; 
        gfloat4 ce = getCenterExtent(index) ;
        printf("  %u :    center %10.3f %10.3f %10.3f   extent %10.3f \n", index, ce.x, ce.y, ce.z, ce.w ); 
    }
}


float* GMergedMesh::getModelToWorldPtr(unsigned int index)
{
    if(index == 0)
    {
        return GMesh::getModelToWorldPtr(0);
    }
    else
    {
        return NULL ; 
    }
}



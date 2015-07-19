#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GBoundaryLib.hh"
#include "GBoundary.hh"
#include <iostream>
#include <iomanip>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo)
{
    GSolid* root = ggeo->getSolid(0);

    unsigned int numMeshes = ggeo->getNumMeshes();
    assert(numMeshes < 500 );


    GMergedMesh* mm = new GMergedMesh( index );

    // 1st pass traversal : counts vertices and faces

    mm->traverse( root, 0, pass_count );  

    // allocate space for flattened arrays

    unsigned int numVertices = mm->getNumVertices();
    mm->setVertices(new gfloat3[numVertices]); 
    mm->setNormals( new gfloat3[numVertices]);
    mm->setColors(  new gfloat3[numVertices]);
    mm->setTexcoords( NULL );  
    mm->setNumColors(numVertices);

    mm->setColor(0.5,0.5,0.5);  // starting point mid-grey, change in traverse 2nd pass

    // consolidate into guint4 
    unsigned int numFaces = mm->getNumFaces();
    mm->setFaces(        new guint3[numFaces]);

    // TODO: consolidate into uint4 with one spare
    mm->setNodes(        new unsigned int[numFaces]);
    mm->setBoundaries(   new unsigned int[numFaces]);
    mm->setSensors(      new unsigned int[numFaces]);

    unsigned int numSolids = mm->getNumSolids();
    mm->setCenterExtent(new gfloat4[numSolids]);
    mm->setTransforms(new float[numSolids*16]);
    mm->setMeshes(new unsigned int[numSolids]);

    // 2nd pass traversal : merge copy GMesh into GMergedMesh 

    mm->traverse( root, 0, pass_merge );  
    mm->updateBounds();


    return mm ;
}



void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GMatrixF* transform = node->getTransform();    

    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GMesh* mesh = solid->getMesh();
    unsigned int meshIndex = mesh->getIndex();
    unsigned int nface = mesh->getNumFaces();
    unsigned int nvert = mesh->getNumVertices();
    gfloat3* vertices = pass == pass_merge ? mesh->getTransformedVertices(*transform) : NULL ;

    bool selected = solid->isSelected();
    if(selected)
    {
        if(pass == pass_count )
        {
            m_num_vertices += nvert ;
            m_num_faces    += nface ; 
        }
        else if(pass == pass_merge )
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
    if(pass == pass_count )
    {
        m_num_solids += 1 ; 
        if(selected) m_num_solids_selected += 1;
    }
    else if( pass == pass_merge ) 
    {
        m_center_extent[m_cur_solid] = GMesh::findCenterExtent(vertices, nvert); // keep track of center_extent of all solids
        m_meshes[m_cur_solid] = meshIndex ; 

        transform->copyTo( m_transforms + m_cur_solid*16 );  

        m_cur_solid += 1 ; 
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}




GMergedMesh* GMergedMesh::load(const char* dir)
{
    GMergedMesh* mm(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        printf("GMergedMesh::load directory %s DOES NOT EXIST \n", dir);
    }
    else
    {
        unsigned int index = 0 ; 
        mm = new GMergedMesh(index);
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



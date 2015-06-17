#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GBoundaryLib.hh"
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo)
{
    GSolid* root = ggeo->getSolid(0);

    GMergedMesh* mm = new GMergedMesh( index );

    // 1st pass traversal : counts vertices and faces

    mm->traverse( root, 0, pass_count );  

    // allocate space for flattened arrays

    mm->setVertices(new gfloat3[mm->getNumVertices()]); // allocate storage 
    mm->setNormals( new gfloat3[mm->getNumVertices()]);
    mm->setColors(  new gfloat3[mm->getNumVertices()]);
    mm->setTexcoords( NULL );  

    mm->setFaces(        new guint3[mm->getNumFaces()]);
    mm->setNodes(        new unsigned int[mm->getNumFaces()]);
    mm->setBoundaries(   new unsigned int[mm->getNumFaces()]);

    mm->setNumColors(mm->getNumVertices());
    mm->setColor(0.5,0.5,0.5);

    mm->setCenterExtent(new gfloat4[mm->getNumSolids()]);

    // 2nd pass traversal : merge GMesh into GMergedMesh

    mm->traverse( root, 0, pass_merge );  
    mm->updateBounds();

    // material/surface properties as function of wavelength collected into wavelengthBuffer

    GBoundaryLib* lib = ggeo->getBoundaryLib();
    lib->createWavelengthAndOpticalBuffers();
    mm->setWavelengthBuffer(lib->getWavelengthBuffer());
    mm->setOpticalBuffer(lib->getOpticalBuffer());

    GPropertyMap<float>* scint = ggeo->findRawMaterial("LiquidScintillator"); // TODO: avoid name specifics at this level
    mm->setReemissionBuffer(lib->createReemissionBuffer(scint));


    return mm ;
}



void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GMatrixF* transform = node->getTransform();    
    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GMesh* mesh = solid->getMesh();
    unsigned int nface = mesh->getNumFaces();
    unsigned int nvert = mesh->getNumVertices();
    gfloat3* vertices = pass == pass_merge ? mesh->getTransformedVertices(*transform) : NULL ;


    bool selected = solid->isSelected();

    if(selected)
    {

        //printf("GMergedMesh::traverse nvert %u nface %u \n", nvert, nface );

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
            unsigned int* node_indices = node->getNodeIndices();
            unsigned int* boundary_indices = node->getBoundaryIndices();

            assert(node_indices);
            assert(boundary_indices);

            for(unsigned int i=0 ; i<nface ; ++i )
            {
                m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
                m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
                m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

                m_nodes[m_cur_faces+i]      = node_indices[i] ;
                m_boundaries[m_cur_faces+i] = boundary_indices[i] ;
            }

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



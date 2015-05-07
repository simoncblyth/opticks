#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"
#include "GSubstanceLib.hh"
#include <iostream>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


GMergedMesh::GMergedMesh(GMergedMesh* other)
       : 
       GMesh(other),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0)

{
}

GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0)
{
} 

GMergedMesh::~GMergedMesh()
{
}

GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo)
{
    GSolid* solid = ggeo->getSolid(0);

    GMergedMesh* mm = new GMergedMesh( index );

    mm->traverse( solid, 0, pass_count );  // 1st pass counts vertices and faces

    mm->setVertices(new gfloat3[mm->getNumVertices()]); // allocate storage 
    mm->setNormals( new gfloat3[mm->getNumVertices()]);
    mm->setColors(  new gfloat3[mm->getNumVertices()]);
    mm->setTexcoords( NULL );  

    mm->setFaces(        new guint3[mm->getNumFaces()]);
    mm->setNodes(        new unsigned int[mm->getNumFaces()]);
    mm->setSubstances(   new unsigned int[mm->getNumFaces()]);

    mm->setNumColors(mm->getNumVertices());
    mm->setColor(0.5,0.5,0.5);

    mm->setCenterExtent(new gfloat4[mm->getNumSolids()]);

    mm->traverse( solid, 0, pass_merge ); // 2nd pass counts merge GMesh into GMergedMesh
    mm->updateBounds();

    GSubstanceLib* lib = ggeo->getSubstanceLib();
    GBuffer* wavelengthBuffer = lib->createWavelengthBuffer();

    mm->setWavelengthBuffer(wavelengthBuffer);
    //mm->dumpWavelengthBuffer(lib->getNumSubstances(), 16, lib->getStandardDomainLength()); 

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
            unsigned int* substance_indices = node->getSubstanceIndices();

            assert(node_indices);
            assert(substance_indices);

            for(unsigned int i=0 ; i<nface ; ++i )
            {
                m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
                m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
                m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

                m_nodes[m_cur_faces+i]      = node_indices[i] ;
                m_substances[m_cur_faces+i] = substance_indices[i] ;
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


void GMergedMesh::dumpWavelengthBuffer(unsigned int numSubstance, unsigned int numProp, unsigned int domainLength)
{
    GBuffer* buffer = getWavelengthBuffer();
    if(!buffer) return ;

    float* data = (float*)buffer->getPointer();

    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert(numElementsTotal == numSubstance*numProp*domainLength);

    std::cout << "GMergedMesh::dumpWavelengthBuffer " 
              << " numSubstance " << numSubstance
              << " numProp " << numProp
              << " domainLength " << domainLength
              << std::endl ; 

    assert(numProp % 4 == 0);

    for(unsigned int isub=0 ; isub < numSubstance ; ++isub )
    {
        unsigned int subOffset = domainLength*numProp*isub ;

        for(unsigned int p=0 ; p < numProp/4 ; ++p ) // property scrunch into float4 is the cause of the gymnastics
        {
             printf("sub %u/%u  prop %u/%u \n", isub, numSubstance, p, numProp/4 );

             unsigned int offset = subOffset + ( p*domainLength*4 ) ;
             for(unsigned int l=0 ; l < 4 ; ++l )
             {
                 for(unsigned int d=0 ; d < domainLength ; ++d )
                 {
                     if(d%5 == 0) printf(" %15.3f", data[offset+d*4+l] );  // too many numbers so display one in every 5
                 }
                 printf("\n");
             }

        }
    }
}





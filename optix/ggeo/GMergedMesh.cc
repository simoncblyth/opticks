#include "GMergedMesh.hh"
#include "GGeo.hh"
#include "GSolid.hh"

GMergedMesh::GMergedMesh(GMergedMesh* other)
       : 
       GMesh(other),
       m_cur_vertices(0),
       m_cur_faces(0)
{
}

GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0)
{
} 

GMergedMesh::~GMergedMesh()
{
}

GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo)
{
    //printf("GMergedMesh::create  %u  \n", index );
    GSolid* solid = ggeo->getSolid(0);

    GMergedMesh* mm = new GMergedMesh( index );

    // 1st pass counts vertices and faces
    mm->traverse( solid, 0, pass_count );  

    // allocate storage 
    mm->setVertices(new gfloat3[mm->getNumVertices()]);
    mm->setNormals( new gfloat3[mm->getNumVertices()]);
    mm->setColors(  new gfloat3[mm->getNumVertices()]);
    mm->setTexcoords( NULL );  

    mm->setFaces(        new guint3[mm->getNumFaces()]);
    mm->setNodes(        new unsigned int[mm->getNumFaces()]);
    mm->setSubstances(   new unsigned int[mm->getNumFaces()]);

    mm->setNumColors(mm->getNumVertices());
    mm->setColor(0.5,0.5,0.5);

    // 2nd pass counts merge GMesh into GMergedMesh
    mm->traverse( solid, 0, pass_merge ); 
    mm->updateBounds();

    return mm ;
}

void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GMatrixF* transform = node->getTransform();    
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    if(solid->isSelected())
    {
        GMesh* mesh = solid->getMesh();
        unsigned int nvert = mesh->getNumVertices();
        unsigned int nface = mesh->getNumFaces();

        //printf("GMergedMesh::traverse nvert %u nface %u \n", nvert, nface );

        if(pass == pass_count )
        {
            m_num_vertices += nvert ;
            m_num_faces    += nface ; 
        }
        else if(pass == pass_merge )
        {
            gfloat3* vertices = mesh->getTransformedVertices(*transform);
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
            unsigned int* nodes = mesh->getNodes();
            unsigned int* substances = mesh->getSubstances();
            for(unsigned int i=0 ; i<nface ; ++i )
            {
                m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
                m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
                m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

                if(nodes && substances)
                { 
                    m_nodes[m_cur_faces+i]      = nodes[i] ;
                    m_substances[m_cur_faces+i] = substances[i] ;
                }
            }

            m_cur_vertices += nvert ;
            m_cur_faces    += nface ;
        }
    } 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}




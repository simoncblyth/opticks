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
       GMesh(index, NULL, 0, NULL, 0 ),
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

    GMergedMesh* mmesh = new GMergedMesh( index );

    mmesh->traverse( solid, 0, pass_count ); 

    mmesh->setVertices(new gfloat3[mmesh->getNumVertices()]);
    mmesh->setColors(  new gfloat3[mmesh->getNumVertices()]);
    mmesh->setFaces(   new guint3[mmesh->getNumFaces()]);

    mmesh->setNumColors(mmesh->getNumVertices());
    mmesh->setColor(0.5,0.5,0.5);

    mmesh->traverse( solid, 0, pass_merge ); 
    mmesh->updateBounds();

    


    return mmesh ;
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
            gfloat3* vertices = mesh->getVertices();
            for(unsigned int i=0 ; i<nvert ; ++i )
            {
                m_vertices[m_cur_vertices+i] = vertices[i] ; 
                m_vertices[m_cur_vertices+i] *= *transform ;
                // fake to mid grey for now
                //m_colors[m_cur_vertices+i].x  = 0.5 ;
                //m_colors[m_cur_vertices+i].y  = 0.5 ;
                //m_colors[m_cur_vertices+i].z  = 0.5 ;
            }

            guint3* faces = mesh->getFaces();
            for(unsigned int i=0 ; i<nface ; ++i )
            {
                m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
                m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
                m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  
            }

            m_cur_vertices += nvert ;
            m_cur_faces    += nface ;
        }
    } 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}




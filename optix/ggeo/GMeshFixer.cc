#include "GMeshFixer.hh"
#include "GMesh.hh"

// brap-
//#include "stringutil.hh"
#include "md5digest.hh"

void GMeshFixer::copyWithoutVertexDuplicates()
{
    mapVertices();
    copyDedupedVertices();
}

void GMeshFixer::mapVertices()
{
    unsigned int num_vertices = m_src->getNumVertices();
    float* vertices = (float*)m_src->getVertices();

    // this should be done separately for each solid, 
    // to retain independance between solids

    m_old2new = new int[num_vertices]; 
    m_new2old = new int[num_vertices];  // cannot be more new vertices than old

    unsigned int vidx = 0 ;         // new de-duped vertex index, into array to be created
    for(unsigned int i=0 ; i < num_vertices ; i++)
    {
        std::string dig = MD5Digest::arraydigest<float>(vertices + 3*i, 3);

        if(m_vtxmap.count(dig) == 0)  // unseen vertex based on digest identity
        {
            m_vtxmap[dig] = vidx ;
            m_new2old[vidx] = i ;
            vidx += 1 ;
        }

        m_old2new[i] = m_vtxmap[dig]  ;
    }

    m_num_deduped_vertices = vidx ; 
}


void GMeshFixer::copyDedupedVertices()
{
    float* vertices  = (float*)m_src->getVertices();
    float* normals   = (float*)m_src->getNormals();
    float* texcoords = (float*)m_src->getTexcoords();
    float* colors    = (float*)m_src->getColors();

    float* dd_vertices  = new float[m_num_deduped_vertices*3] ;
    float* dd_normals   = normals   ? new float[m_num_deduped_vertices*3] : NULL ;
    float* dd_texcoords = texcoords ? new float[m_num_deduped_vertices*2] : NULL ;
    float* dd_colors    = colors    ? new float[m_num_deduped_vertices*3] : NULL ;

    for(unsigned int n=0 ; n < m_num_deduped_vertices ; ++n )
    {
        int o = m_new2old[n] ;

        for(unsigned int j=0 ; j < 3 ; j++)
           *(dd_vertices + n*3 + j ) = *(vertices + 3*o + j) ;

        if(normals)
            for(unsigned int j=0 ; j < 3 ; j++)
                *(dd_normals  + n*3 + j ) = *(normals  + 3*o + j) ;

        if(colors)
            for(unsigned int j=0 ; j < 3 ; j++)
                *(dd_colors   + n*3 + j ) = *(colors   + 3*o + j) ;

        if(texcoords)
            for(unsigned int j=0 ; j < 2 ; j++)
                *(dd_texcoords + n*2 + j ) = *(texcoords + 2*o + j) ;
    }


    unsigned int num_faces = m_src->getNumFaces() ;   
    unsigned int* faces    = (unsigned int*)m_src->getFaces();
    unsigned int* dd_faces = new unsigned int[num_faces*3] ;

    for(unsigned int f=0 ; f < num_faces ; ++f )
    {
       for(unsigned int j=0 ; j < 3 ; j++)
       {
           int oj = *(faces + 3*f + j) ;
           *(dd_faces + f*3 + j ) = m_old2new[oj] ;
       }
    }


    m_dst = new GMesh( m_src->getIndex(), 
                       (gfloat3*)dd_vertices , m_num_deduped_vertices, 
                       (guint3*)dd_faces,  num_faces  ,
                       (gfloat3*)dd_normals, 
                       (gfloat2*)dd_texcoords ); 

    m_dst->setColors( (gfloat3*)dd_colors );
}




#include "GMesh.hh"
#include "GBuffer.hh"
#include "stdio.h"
#include "assert.h"
#include <algorithm>

GMesh::GMesh(GMesh* other) 
     :
     m_index(other->getIndex()),
     m_vertices(other->getVertices()),
     m_vertices_buffer(other->getVerticesBuffer()),
     m_num_vertices(other->getNumVertices()),
     m_faces(other->getFaces()),
     m_indices_buffer(other->getIndicesBuffer()),
     m_num_faces(other->getNumFaces()),
     m_colors(other->getColors()),
     m_colors_buffer(other->getColorsBuffer()),
     m_texcoords(other->getTexcoords()),
     m_texcoords_buffer(other->getTexcoordsBuffer()),
     m_num_colors(other->getNumColors()),
     m_normals(other->getNormals()),
     m_normals_buffer(other->getNormalsBuffer()),
     m_nodes(other->getNodes()),
     m_nodes_buffer(other->getNodesBuffer()),
     m_substances(other->getSubstances()),
     m_substances_buffer(other->getSubstancesBuffer()),
     GDrawable()
{
   updateBounds();
}


int GMesh::g_instance_count = 0 ;

GMesh::GMesh(unsigned int index, 
             gfloat3* vertices, 
             unsigned int num_vertices, 
             guint3* faces, 
             unsigned int num_faces, 
             gfloat3* normals, 
             gfloat2* texcoords
            ) 
        :
      m_index(index),
      m_vertices(NULL),
      m_vertices_buffer(NULL),
      m_num_vertices(num_vertices), 
      m_faces(NULL),
      m_indices_buffer(NULL),
      m_nodes(NULL),
      m_nodes_buffer(NULL),
      m_substances(NULL),
      m_substances_buffer(NULL),
      m_num_faces(num_faces),
      m_colors(NULL),
      m_colors_buffer(NULL),
      m_texcoords(NULL),
      m_texcoords_buffer(NULL),
      m_low(NULL),
      m_high(NULL),
      m_dimensions(NULL),
      m_center(NULL),
      m_model_to_world(NULL),
      m_extent(0.f),
      m_num_colors(num_vertices),  // tie num_colors to num_vertices
      m_normals(NULL),
      m_normals_buffer(NULL),
      GDrawable()
{
   // not yet taking ownership, depends on continued existance of data source 

   g_instance_count += 1 ; 

   printf("GMesh::GMesh  index %d g_instance_count %d \n", index, g_instance_count );

   setVertices(vertices);
   setFaces(faces);
   setNormals(normals);
   setTexcoords(texcoords);
   updateBounds();
}


unsigned int GMesh::getIndex()
{
    return m_index ; 
}
unsigned int GMesh::getNumVertices()
{
    return m_num_vertices ; 
}
unsigned int GMesh::getNumColors()
{
    return m_num_colors ;   
}
unsigned int GMesh::getNumFaces()
{
    return m_num_faces ; 
}


void GMesh::setNumColors(unsigned int num_colors)
{
   m_num_colors = num_colors ;
}






gfloat3* GMesh::getLow()
{
    return m_low ;
}
gfloat3* GMesh::getHigh()
{
    return m_high ;
}
gfloat3* GMesh::getDimensions()
{
    return m_dimensions ; 
}

GMatrix<float>* GMesh::getModelToWorld()
{
    return m_model_to_world ; 
}





gfloat3* GMesh::getVertices()
{
    return m_vertices ;
}
gfloat3* GMesh::getNormals()
{
    return m_normals ;
}

gfloat3* GMesh::getColors()
{
    return m_colors ;
}
gfloat2* GMesh::getTexcoords()
{
    return m_texcoords ;
}


guint3*  GMesh::getFaces()
{
    return m_faces ;
}
unsigned int* GMesh::getNodes()
{
    return m_nodes ;
}
unsigned int* GMesh::getSubstances()
{
    return m_substances ;
}



GBuffer* GMesh::getVerticesBuffer()
{
    return m_vertices_buffer ;
}
GBuffer* GMesh::getNormalsBuffer()
{
    return m_normals_buffer ;
}



GBuffer* GMesh::getColorsBuffer()
{
    return m_colors_buffer ;
}
GBuffer* GMesh::getTexcoordsBuffer()
{
    return m_texcoords_buffer ;
}



GBuffer*  GMesh::getIndicesBuffer()
{
    return m_indices_buffer ;
}
GBuffer*  GMesh::getNodesBuffer()
{
    return m_nodes_buffer ;
}
GBuffer*  GMesh::getSubstancesBuffer()
{
    return m_substances_buffer ;
}





GBuffer*  GMesh::getModelToWorldBuffer()
{
    return (GBuffer*)m_model_to_world ;
}






void GMesh::setVertices(gfloat3* vertices)
{
    m_vertices = vertices ;
    m_vertices_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_vertices, sizeof(gfloat3), 3 ) ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}
void GMesh::setNormals(gfloat3* normals)
{
    m_normals = normals ;
    m_normals_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_normals, sizeof(gfloat3), 3 ) ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}

void GMesh::setFaces(guint3* faces)
{
    m_faces = faces ;
    m_indices_buffer = new GBuffer( sizeof(guint3)*m_num_faces, (void*)m_faces, sizeof(guint3)/3, 1 ) ;
    assert(sizeof(guint3) == sizeof(unsigned int)*3);
}
void GMesh::setNodes(unsigned int* nodes)
{
    m_nodes = nodes ;
    m_nodes_buffer = new GBuffer( sizeof(unsigned int)*m_num_faces, (void*)m_nodes, sizeof(unsigned int), 1 ) ;
    assert(sizeof(unsigned int) == sizeof(unsigned int)*1);
}
void GMesh::setSubstances(unsigned int* substances)
{
    m_substances = substances ;
    m_substances_buffer = new GBuffer( sizeof(unsigned int)*m_num_faces, (void*)m_substances, sizeof(unsigned int), 1 ) ;
    assert(sizeof(unsigned int) == sizeof(unsigned int)*1);
}


void GMesh::setColors(gfloat3* colors)
{
    m_colors = colors ;
    m_colors_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_colors, sizeof(gfloat3), 3  ) ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}

void GMesh::setTexcoords(gfloat2* texcoords)
{
    if(!texcoords) return ;
    m_texcoords = texcoords ;
    m_texcoords_buffer = new GBuffer( sizeof(gfloat2)*m_num_vertices, (void*)m_texcoords, sizeof(gfloat2), 2  ) ;
    assert(sizeof(gfloat2) == sizeof(float)*2);
}





void GMesh::setColor(float r, float g, float b)
{
    assert(m_num_colors == m_num_vertices);
    if(!m_colors)
    {
        setColors(new gfloat3[m_num_colors]);
    }
    for(unsigned int i=0 ; i<m_num_colors ; ++i )
    {
        m_colors[i].x  = r ;
        m_colors[i].y  = g ;
        m_colors[i].z  = b ;
    }
}


void GMesh::setLow(gfloat3* low)
{
    m_low = low ;
}
void GMesh::setHigh(gfloat3* high)
{
    m_high = high ;
}

bool GMesh::hasTexcoords()
{
    return m_texcoords != NULL ;
}


GMesh::~GMesh()
{
}
void GMesh::Dump(const char* msg, unsigned int nmax)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
    {
        gfloat3& vtx = m_vertices[i] ;
        printf(" vtx %5u  %10.3f %10.3f %10.3f \n", i, vtx.x, vtx.y, vtx.z );
    } 

    for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
    {
        gfloat3& nrm = m_normals[i] ;
        printf(" nrm %5u  %10.3f %10.3f %10.3f \n", i, nrm.x, nrm.y, nrm.z );
    } 


    if(hasTexcoords())
    { 
        for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
        {
            gfloat2& tex = m_texcoords[i] ;
            printf(" tex %5u  %10.3f %10.3f  \n", i, tex.u, tex.v );
        } 
    }

    for(unsigned int i=0 ; i < std::min(nmax,m_num_colors) ; i++)
    {
        gfloat3& col = m_colors[i] ;
        printf(" col %5u  %10.3f %10.3f %10.3f \n", i, col.x, col.y, col.z );
    } 


    for(unsigned int i=0 ; i < std::min(nmax,m_num_faces) ; i++)
    {
        guint3& fac = m_faces[i] ;
        printf(" fac %5u  %5u %5u %5u \n", i, fac.x, fac.y, fac.z );
    } 

    for(unsigned int i=0 ; i < std::min(nmax,m_num_faces) ; i++)
    {
        unsigned int& node = m_nodes[i] ;
        unsigned int& substance = m_substances[i] ;
        printf(" fac %5u  node %5u substance %5u  \n", i, node, substance );
    } 




}


void GMesh::Summary(const char* msg)
{
   printf("%s idx %u vx %u fc %u \n",
      msg, 
      m_index, 
      m_num_vertices, 
      m_num_faces);

   printf("%10s %10.3f %10.3f %10.3f\n",
         "low",
         m_low->x,
         m_low->y,
         m_low->z);

   printf("%10s %10.3f %10.3f %10.3f\n", 
          "high",
          m_high->x,
          m_high->y,
          m_high->z);

   printf("%10s %10.3f %10.3f %10.3f extent %10.3f\n", 
          "dimen",
          m_dimensions->x,
          m_dimensions->y,
          m_dimensions->z,
          m_extent);

   printf("%10s %10.3f %10.3f %10.3f\n", 
          "center",
          m_center->x,
          m_center->y,
          m_center->z);

   m_model_to_world->Summary(msg);
}


float GMesh::getExtent()
{
     return m_extent ;  
}


void GMesh::updateBounds()
{
    gfloat3  low( 1e10f, 1e10f, 1e10f);
    gfloat3 high( -1e10f, -1e10f, -1e10f);

    for( unsigned int i = 0; i < m_num_vertices ;++i )
    {
        gfloat3& v = m_vertices[i];

        low.x = std::min( low.x, v.x);
        low.y = std::min( low.y, v.y);
        low.z = std::min( low.z, v.z);

        high.x = std::max( high.x, v.x);
        high.y = std::max( high.y, v.y);
        high.z = std::max( high.z, v.z);
    }

    m_low = new gfloat3(low.x, low.y, low.z) ;
    m_high = new gfloat3(high.x, high.y, high.z);

    m_dimensions = new gfloat3(high.x - low.x, high.y - low.y, high.z - low.z );
    m_center     = new gfloat3((high.x + low.x)/2.0f, (high.y + low.y)/2.0f , (high.z + low.z)/2.0f );
    m_extent = 0.f ;
    m_extent = std::max( m_dimensions->x , m_extent );
    m_extent = std::max( m_dimensions->y , m_extent );
    m_extent = std::max( m_dimensions->z , m_extent );
    m_extent = m_extent / 2.0f ;         
    //
    // extent is half the maximal dimension 
    // 
    // model coordinates definition
    //      all vertices are contained within model coordinates box  (-1:1,-1:1,-1:1) 
    //      model coordinates origin (0,0,0) corresponds to world coordinates  m_center
    //  
    // world -> model
    //        * translate by -m_center 
    //        * scale by 1/m_extent
    //
    //  model -> world
    //        * scale by m_extent
    //        * translate by m_center
    //

    m_model_to_world = new GMatrix<float>( m_center->x, m_center->y, m_center->z, m_extent );
}



void GMesh::updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform)
{
    if(m_low && m_high)
    {   
        gfloat3 mlow(*m_low) ; 
        gfloat3 mhigh(*m_high) ; 

        mlow  *= transform ; 
        mhigh *= transform ; 

        low.x = std::min( low.x, mlow.x);
        low.y = std::min( low.y, mlow.y);
        low.z = std::min( low.z, mlow.z);

        high.x = std::max( high.x, mhigh.x);
        high.y = std::max( high.y, mhigh.y);
        high.z = std::max( high.z, mhigh.z);
   }
}



gfloat3* GMesh::getTransformedVertices(GMatrixF& transform )
{
     gfloat3* vertices = new gfloat3[m_num_vertices];
     for(unsigned int i = 0; i < m_num_vertices; i++)
     {  
         vertices[i].x = m_vertices[i].x ;   
         vertices[i].y = m_vertices[i].y ;   
         vertices[i].z = m_vertices[i].z ;   

         vertices[i] *= transform ;
     }   
     return vertices ;
}

gfloat3* GMesh::getTransformedNormals(GMatrixF& transform )
{
     gfloat3* normals = new gfloat3[m_num_vertices];
     for(unsigned int i = 0; i < m_num_vertices; i++)
     {  
         gfloat4 nrm(m_normals[i], 0.);   // w=0 as direction, not position 

         nrm *= transform ; 
         // NB should be transpose of inverse, 
         // (so only OK if orthogonal, that means rotations only no scaling)

         normals[i].x = nrm.x ; 
         normals[i].y = nrm.y ; 
         normals[i].z = nrm.z ; 

     }   
     return normals ;
}



void GMesh::updateDistinctSubstances()
{
    for(unsigned int i=0 ; i < getNumFaces() ; i++)
    {
        unsigned int index = m_substances[i] ;
        if(std::count(m_distinct_substances.begin(), m_distinct_substances.end(), index ) == 0) m_distinct_substances.push_back(index);
    }  
    std::sort( m_distinct_substances.begin(), m_distinct_substances.end() );
}
 
std::vector<unsigned int>& GMesh::getDistinctSubstances()
{
    if(m_distinct_substances.size()==0) updateDistinctSubstances();
    return m_distinct_substances ;
}








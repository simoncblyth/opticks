#include <algorithm>
#include <iostream>

// npy-
#include "NGLM.hpp"
#include "GLMPrint.hpp"

#include "GVector.hh"
#include "GMatrix.hh"


GMatrix<float>* GMesh_make_model_to_world(const gfloat3& low, const gfloat3& high)
{
    GMatrix<float>* m_model_to_world ; 


   // from GMesh::updateBounds 
    //gfloat3 *m_low, *m_high ;
    //m_low = new gfloat3(low.x, low.y, low.z) ;
    //m_high = new gfloat3(high.x, high.y, high.z);

    gfloat3  *m_dimensions, *m_center ;   // * belongs to the var, not the type
    m_dimensions = new gfloat3(high.x - low.x, high.y - low.y, high.z - low.z );
    m_center     = new gfloat3((high.x + low.x)/2.0f, (high.y + low.y)/2.0f , (high.z + low.z)/2.0f );

    float m_extent ; 
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

    return m_model_to_world ; 
}


glm::mat4 GLM_make_model_to_world(const gfloat3& low, const gfloat3& high, bool correct=true)
{

    //glm::vec3 *m_low, *m_high ; 
    //m_low = new glm::vec3(low.x, low.y, low.z);
    //m_high = new glm::vec3(high.x, high.y, high.z);

    glm::vec3 *m_dimensions, *m_center ;   // * belongs to the var, not the type
    m_dimensions = new glm::vec3(high.x - low.x, high.y - low.y, high.z - low.z );
    m_center     = new glm::vec3((high.x + low.x)/2.0f, (high.y + low.y)/2.0f , (high.z + low.z)/2.0f );

    float m_extent ; 
    m_extent = 0.f ;
    m_extent = std::max( m_dimensions->x , m_extent );
    m_extent = std::max( m_dimensions->y , m_extent );
    m_extent = std::max( m_dimensions->z , m_extent );
    m_extent = m_extent / 2.0f ;    

    glm::vec3 s(m_extent);
    glm::vec3 t(*m_center);

    if(correct)
        return glm::scale(glm::translate(glm::mat4(1.0), t), s); 
    else
        return glm::translate(glm::scale(glm::mat4(1.0), s), t); 
}



void test_model_to_world()
{
    // finding GLM approach that yields M2W matrices consistent with GMatrix

    gfloat3 low(-1100.f,-1100.f,-1100.f);
    gfloat3 high(2100.f,2100.f,2100.f);

    GMatrix<float>* m = GMesh_make_model_to_world(low, high) ;
    m->Summary("GMatrix::Summary presentation puts translation in right column");
    float* gmatrix_m2w = (float*)m->getPointer();
    print(gmatrix_m2w, "gmatrix_m2w : GMatrix::getPointer() provides raw floats in OpenGL conventional order, translation within last 4 ");

    glm::mat4 gmatrix_m2w_as_mat4 = glm::make_mat4(gmatrix_m2w);
    print(gmatrix_m2w_as_mat4, "gmatrix_m2w_as_mat4: Common::print(glm::mat4&,) presents in transposed order (for familiarity?)");



    glm::mat4 mm = GLM_make_model_to_world(low, high);

    print(mm, "mm Common::print(glm::mat4&) presents in transposed order...");

    std::cout << "glm::to_string(mm) " << glm::to_string(mm) << std::endl ;


    float* glm_m2w = glm::value_ptr(mm) ;
    print(glm_m2w, "glm_m2w : glm::value_ptr(mm) what lies beneath follows OpenGL convention");


    for(unsigned int i=0 ; i < 16 ; ++i )
    {
        printf(" %2u  %15f %15f \n", i, gmatrix_m2w[i], glm_m2w[i] );
        assert( gmatrix_m2w[i] == glm_m2w[i]  );
    }
}


int main()
{
    test_model_to_world();
    return 0 ;
}




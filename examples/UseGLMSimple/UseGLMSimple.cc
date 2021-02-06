
#include <iostream>

#include <vector>
#include <glm/mat4x4.hpp> 
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

float unsigned_as_float( unsigned u )
{
    union { unsigned u; int i; float f; } uif ;  
    uif.u = u  ; 
    return uif.f ; 
}

unsigned float_as_unsigned( float f )
{
    union { unsigned u; int i; float f; } uif ;  
    uif.f = f  ; 
    return uif.u ; 
}

void test_planting_uint_in_mat4()
{
    glm::mat4 mat(1.0f); 

    std::cout << glm::to_string(mat) << std::endl ;  

    unsigned a = 42u ; 
    unsigned b = 43u ; 
    unsigned c = 44u ; 
    unsigned d = 45u ; 

    mat[0][3] = unsigned_as_float(a) ; 
    mat[1][3] = unsigned_as_float(b) ; 
    mat[2][3] = unsigned_as_float(c) ; 
    mat[3][3] = unsigned_as_float(d) ; 

    std::cout << glm::to_string(mat) << std::endl ;  

    glm::mat4 imat = glm::transpose(mat); 
    std::cout << glm::to_string(imat) << std::endl ;  

    glm::uvec4 idv ; 
    memcpy( glm::value_ptr(idv), &imat[3], 4*sizeof(float) ) ;

    std::cout << glm::to_string(idv) << std::endl ;  

    assert( idv.x == a ); 
    assert( idv.y == b ); 
    assert( idv.z == c ); 
    assert( idv.w == d ); 
}

void test_vector_of_mat4()
{
    std::vector<glm::mat4> vmat ; 

    unsigned num = 10 ; 

    for(unsigned i=0 ; i < num ; i++)
    {
        glm::mat4 mat(1.0f); 
        mat[3][3] = unsigned_as_float(i+100) ;
        vmat.push_back(mat); 
    }    

    
    char* data = (char*)vmat.data() ;
    unsigned num_bytes = vmat.size()*sizeof(glm::mat4) ; 
    unsigned num_mat = num_bytes / sizeof(glm::mat4); 
    

    assert( num == num_mat ) ;  
    std::cout << " num_mat " << num_mat << std::endl ; 


    for(unsigned i=0 ; i < num_mat ; i++)
    {
        glm::mat4 mat(1.0f);
        memcpy( glm::value_ptr(mat), data+i*sizeof(glm::mat4), sizeof(glm::mat4) );   

        unsigned u = float_as_unsigned( mat[3][3] ); 

        std::cout << i << " " << u << " " << glm::to_string(mat) << std::endl ;  
    }
}


int main(int argc, char** argv)
{
    //test_planting_uint_in_mat4(); 

    test_vector_of_mat4(); 

    return 0 ; 
}





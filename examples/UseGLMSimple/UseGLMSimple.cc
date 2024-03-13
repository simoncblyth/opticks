
#include <cstdint>
#include <sstream>
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


template<typename U, typename I, typename F>
union TV4 
{ 
    glm::tvec4<U> u; 
    glm::tvec4<I>  i; 
    glm::tvec4<F>    f; 

    std::string desc() const ; 
}; 


template<typename U, typename I, typename F>
inline std::string TV4<U,I,F>::desc() const
{
    std::stringstream ss ; 
    ss << "tv4::desc"
        << std::endl 
        << " u " << glm::to_string( u )
        << std::endl 
        << " i " << glm::to_string( i )
        << std::endl 
        << " f " << glm::to_string( f )
        << std::endl 
        ;
    std::string str = ss.str(); 
    return str ; 
}


void test_tv4_union()
{
    typedef TV4<uint32_t,int32_t,float> tv4_32 ; 
    tv4_32 q0 ; 
    q0.u = {1, 2, 3, 4} ; 
    std::cout << q0.desc(); 

    q0.f.w = 42.f ; 
    std::cout << q0.desc(); 

    typedef TV4<uint64_t,int64_t,double> tv4_64 ; 
    tv4_64 q1 ; 
    q1.u = {1, 2, 3, 4} ; 
    std::cout << q1.desc(); 

    q1.f.w = 42. ; 
    std::cout << q1.desc(); 
}


int main(int argc, char** argv)
{
    /*
    test_planting_uint_in_mat4(); 
    test_vector_of_mat4(); 
    */
    test_tv4_union(); 

    return 0 ; 
}





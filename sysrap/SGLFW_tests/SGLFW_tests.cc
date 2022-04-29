#include <iostream>
#include <iomanip>
#include "SGLFW.hh"

void test_SGLFW_GLenum(const char* name)
{
    GLenum type = SGLFW_GLenum::Type(name); 
    const char* name2 = SGLFW_GLenum::Name(type); 
    std::cout 
        << " name " << std::setw(20) << name 
        << " type " << std::setw(5) << type 
        << " name2 " << std::setw(20) << name2
        << std::endl 
        ; 
    assert( strcmp(name, name2) == 0 ); 
}

void test_SGLFW_GLenum()
{
    test_SGLFW_GLenum("GL_BYTE"); 
    test_SGLFW_GLenum("GL_UNSIGNED_BYTE"); 
    test_SGLFW_GLenum("GL_SHORT"); 
    test_SGLFW_GLenum("GL_UNSIGNED_SHORT"); 
    test_SGLFW_GLenum("GL_INT"); 
    test_SGLFW_GLenum("GL_UNSIGNED_INT"); 
    test_SGLFW_GLenum("GL_HALF_FLOAT"); 
    test_SGLFW_GLenum("GL_FLOAT"); 
    test_SGLFW_GLenum("GL_DOUBLE"); 
}

void test_SGLFW_Attribute(const char* name, const char* spec)
{
    SGLFW_Attribute att(name, spec); 
    std::cout << att.desc() << std::endl ; 
}

void test_SGLFW_Attribute()
{
    test_SGLFW_Attribute("rpos","4,GL_FLOAT,GL_FALSE,64,0,false") ; 
}

/**


    // array attribute : connecting values from the array with attribute symbol in the shader program 
    GLint rpos_location                = glGetAttribLocation( sglfw.program, "rpos");                  SGLFW::check(__FILE__, __LINE__);
    std::cout << " rpos_location " << rpos_location << std::endl ;

    GLsizei stride = sizeof(float)*4*4 ;
    const void* rpos_offset = (void*)(sizeof(float)*0) ;   // pos
    if( rpos_location > -1 )
    {
        glEnableVertexAttribArray(rpos_location);                                              SGLFW::check(__FILE__, __LINE__);
        glVertexAttribPointer(rpos_location, 4, GL_FLOAT, GL_FALSE, stride, rpos_offset );     SGLFW::check(__FILE__, __LINE__);
    }




**/




int main()
{
    test_SGLFW_GLenum(); 
    test_SGLFW_Attribute(); 
    return 0 ; 
}

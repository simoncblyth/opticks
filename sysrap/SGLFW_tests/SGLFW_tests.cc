#include <iostream>
#include <iomanip>
#include "SGLFW.hh"

void test_SGLFW_Type(const char* name)
{
    GLenum type = SGLFW_Type::Type(name); 
    const char* name2 = SGLFW_Type::Name(type); 
    std::cout 
        << " name " << std::setw(20) << name 
        << " type " << std::setw(5) << type 
        << " name2 " << std::setw(20) << name2
        << std::endl 
        ; 
    assert( strcmp(name, name2) == 0 ); 
}

void test_SGLFW_Type()
{
    test_SGLFW_Type("GL_BYTE"); 
    test_SGLFW_Type("GL_UNSIGNED_BYTE"); 
    test_SGLFW_Type("GL_SHORT"); 
    test_SGLFW_Type("GL_UNSIGNED_SHORT"); 
    test_SGLFW_Type("GL_INT"); 
    test_SGLFW_Type("GL_UNSIGNED_INT"); 
    test_SGLFW_Type("GL_HALF_FLOAT"); 
    test_SGLFW_Type("GL_FLOAT"); 
    test_SGLFW_Type("GL_DOUBLE"); 
}

void test_SGLFW_Attribute(const char* name_spec)
{
    SGLFW_Attribute att(name_spec); 
    std::cout << att.desc() << std::endl ; 
}

void test_SGLFW_Attribute()
{
    test_SGLFW_Attribute("rpos:hello") ; 
}


int main()
{
    test_SGLFW_Type(); 
    test_SGLFW_Attribute(); 
    return 0 ; 
}

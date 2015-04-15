#include "NPY.hpp"
#include "numpy.hpp"
#include <iostream>

std::string path_(const char* typ, const char* tag)
{
    char* TYP = strdup(typ);
    char* p = TYP ;
    while(*p)
    {
       if( *p >= 'a' && *p <= 'z') *p += 'A' - 'a' ;
       p++ ; 
    } 

    char envvar[64];
    snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
    free(TYP); 

    char* tmpl = getenv(envvar) ;
    if(!tmpl) return "missing-template-envvar" ; 
    
    char path[256];
    snprintf(path, 256, tmpl, tag );

    return path ;   
}


void test_ctor()
{
    std::vector<int> shape = {2,2} ;
    std::vector<float> data = {1.f,2.f,3.f,4.f}  ;
    std::string metadata = "{}";

    NPY npy(shape,data,metadata) ;
    std::cout << npy.description("npy") << std::endl ; 
}

void test_path()
{
    std::string path = path_("cerenkov", "1");
    std::cout << path << std::endl ; 
}

void test_load()
{
    std::string path = path_("cerenkov", "1");
    std::vector<int> shape ;
    std::vector<float> data ;

    aoba::LoadArrayFromNumpy<float>(path, shape, data );

    std::string metadata = "{}";
    NPY npy(shape,data,metadata) ;
    std::cout << npy.description("npy") << std::endl ; 
}



int main()
{
    test_load();
    return 0 ;
}

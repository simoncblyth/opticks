// op --idpath

#include "Opticks.hh"
#include "OpticksResource.hh"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv) ;
    OpticksResource res(&ok) ;  
    printf("%s\n",res.getIdPath());
    return 0 ; 
}

// ./SName_test.sh
#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>

#include "SName.h"

void test_args(int argc, char** argv)
{
    const char* name = argv[0] ;     
    char* sname = strdup(name); 

    int nj = int(strlen(sname)) ; 
    
    for(int j=0 ; j < nj ; j++) std::cout << std::setw(2) << j << " " << sname[j] << std::endl  ; 
    std::cout << std::endl ; 


    for(int j=0 ; j < nj ; j++) 
    {
        sname[nj-1-j] = '\0' ; 
        std::cout 
             << std::setw(3) << strlen(sname)
             << " : "
             << sname 
             << std::endl 
             ; 
    }
}

void test_Load()
{
    const char* idp = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/meshname.txt" ; 
    SName* id = SName::Load(idp); 
    std::cout << id->detail() << std::endl ; 
}

void test_GEOMLoad()
{
    SName* id = SName::GEOMLoad(); 
    std::cout 
        << "test_GEOMLoad"
        << std::endl 
        << "SName* id = SName::GEOMLoad() ; id->detail() " 
        << std::endl 
        << id->detail()
        << std::endl 
        ; 
}

int main(int argc, char** argv)
{
    /*
    test_args(argc, argv); 
    test_Load(); 
    */
    test_GEOMLoad(); 

    return 0 ; 
}

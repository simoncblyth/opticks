// name=SName_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 
#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>

int main(int argc, char** argv)
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

    return 0 ; 

}

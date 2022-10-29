// name=pathtype_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int pathtype(const char* path)
{
    int rc = -1 ; 
	struct stat st ;
    if(0 == stat(path, &st))
    {
        if(     S_ISDIR(st.st_mode)) rc = 1 ; 
        else if(S_ISREG(st.st_mode)) rc = 2 ;  
        else                         rc = 3 ; 
    }
    return rc ;  
}



const char* PATHS = R"(
/tmp
/tmp/hello
/tmp/hello/red
/tmp/hello/green
/tmp/hello/blue
/tmp/hello/blue/world
)" ; 

int main(int argc, char** argv)
{
    std::stringstream ss(PATHS); 
    std::string path ; 
    while(std::getline(ss, path))
    {
        int pt = pathtype(path.c_str()); 
        std::cout << " pt " << std::setw(2) << pt  << " path [" << path << "]" << std::endl ;  
    } 
    return 0 ; 
}

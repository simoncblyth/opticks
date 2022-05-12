// export OPTICKS_RANDOM_SEQPATH=/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000
// name=dirent ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include "dirent.h"


void ListDir(std::vector<std::string>& names, const char* path, const char* ext )
{
    DIR* dir = opendir(path) ;
    if(!dir) return ; 
    struct dirent* entry ;
    while ((entry = readdir(dir)) != nullptr) 
    {
        const char* name = entry->d_name ; 
        if(strlen(name) > strlen(ext) && strcmp(name + strlen(name) - strlen(ext), ext)==0)
        {
            //std::cout << name << std::endl;  
            names.push_back(name); 
        }
    }
    closedir (dir);
    std::sort( names.begin(), names.end() ); 
}

std::string Desc(const std::vector<std::string>& names)
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}




void test_0()
{
    //const char* path = "c:\\src\\" ; 
    const char* path = "/private/tmp" ; 

    DIR* dir = opendir(path) ;
    if( dir == NULL  )
    {
        perror ("");
        return ;
    }

    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) 
    {
        printf("%s\n", ent->d_name);
    }

    closedir (dir);
}

void test_1()
{
    const char* path = getenv("OPTICKS_RANDOM_SEQPATH"); 
    const char* ext = ".npy"; 
    std::vector<std::string> names ; 
    ListDir(names, path, ext);  
    std::cout << Desc(names) << std::endl ; 

}




int main(int argc, char** argv)
{
    //test_0(); 
    test_1(); 

    return 0 ; 
}


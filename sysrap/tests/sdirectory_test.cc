// name=sdirectory_test ; gcc $name.cc -I.. -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>

#include "sdirectory.h"
#include "spath.h"
#include "sstr.h"


void test_MakeDirs()
{
    const char* dirpath_ = "/tmp/$USER/opticks/red/green/blue/cyan/purple/puce" ; 
    const char* dirpath = spath::Resolve(dirpath_); 

    int rc = sdirectory::MakeDirs(dirpath, 0 ); 
    std::cout << " dirpath " << dirpath << " rc " << rc << std::endl ; 
}

void test_MakeDirsForFile()
{
    const char* filepath_ = "/tmp/$USER/opticks/some/deep/dir/for/a/file.txt" ; 
    const char* filepath = spath::Resolve(filepath_); 

    int rc = sdirectory::MakeDirsForFile(filepath, 0 ); 
    std::cout << " filepath " << filepath << " rc " << rc << std::endl ; 

    sstr::Write(filepath, "test_MakeDirsForFile\n"); 
}


int main(int argc, char** argv)
{
    /*
    test_MakeDirs(); 
    */
    test_MakeDirsForFile(); 

    return 0 ; 
}

/**
sdirectory_test.cc
===================

~/o/sysrap/tests/sdirectory_test.sh 

**/

#include <cassert>
#include <iostream>

#include "sdirectory.h"
#include "spath.h"
#include "sstr.h"
#include "ssys.h"


struct sdirectory_test
{
    static int MakeDirs();
    static int MakeDirsForFile();
    static int DirList();
    static int Main();
};


int sdirectory_test::MakeDirs()
{
    const char* dirpath_ = "/tmp/$USER/opticks/red/green/blue/cyan/purple/puce" ; 
    const char* dirpath = spath::Resolve(dirpath_); 

    int rc = sdirectory::MakeDirs(dirpath, 0 ); 
    std::cout << " dirpath " << dirpath << " rc " << rc << std::endl ; 

    return 0 ; 
}

int sdirectory_test::MakeDirsForFile()
{
    const char* filepath_ = "/tmp/$USER/opticks/some/deep/dir/for/a/file.txt" ; 
    const char* filepath = spath::Resolve(filepath_); 

    int rc = sdirectory::MakeDirsForFile(filepath, 0 ); 
    std::cout << " filepath " << filepath << " rc " << rc << std::endl ; 

    sstr::Write(filepath, "test_MakeDirsForFile\n"); 
    return 0 ; 
}

int sdirectory_test::DirList()
{
    std::vector<std::string> names ; 
    
    const char* path = spath::Resolve("${RNGDir:-$HOME/.opticks/rngcache/RNG}") ; 
    const char* pfx = "QCurandState_" ; 
    const char* ext = ".bin" ; 
    sdirectory::DirList(names, path, pfx, ext ); 

    std::cout << "names.size " << names.size() << "\n" ; 
    for(int i=0 ; i < int(names.size()) ; i++) std::cout << names[i] << "\n" ; 
    return 0 ; 
}



int sdirectory_test::Main()
{ 
    int rc = 0 ; 
    const char* TEST = ssys::getenvvar("TEST", "DirList") ; 

    if(strcmp(TEST, "MakeDirs") == 0 )        rc += MakeDirs(); 
    if(strcmp(TEST, "MakeDirsForFile") == 0 ) rc += MakeDirsForFile(); 
    if(strcmp(TEST, "DirList") == 0 )         rc += DirList(); 

    return rc ; 
}


int main(){ return sdirectory_test::Main() ; }

// ./spath_test.sh

#include <cassert>
#include <iostream>
#include <iomanip>
#include "spath.h"


void test_ResolvePath()
{
    std::vector<std::string> specs = {
        "$HOME/hello.npy", 
        "$HOME", 
        "$HOME/red/green/blue$HOME",
        "$HOME/red/green/blue$HOME/cyan/magenta"
         } ; 

    for(unsigned i=0 ; i < specs.size() ; i++)
    { 
        const char* spec = specs[i].c_str(); 
        const char* path = spath::ResolvePath(spec); 
        std::cout 
            << " spec " << spec << std::endl  
            << " path " << path << std::endl
            << std::endl 
            ;
    }
}

void test_Resolve()
{
    const char* path0 = spath::Resolve("$HOME", "subdir", "another", "name.npy" ) ;  
    const char* path1 = spath::Resolve("$HOME/subdir/another/name.npy" ) ;  

    std::cout << " path0 [" << path0 << "]" << std::endl ;   
    std::cout << " path1 [" << path1 << "]" << std::endl ;   
}

void test_Exists()
{
    std::vector<std::string> specs = {"$HOME/hello.npy", "$HOME", "$OPTICKS_HOME/sysrap/tests/spath_test.cc" } ; 
    for(unsigned i=0 ; i < specs.size() ; i++)
    { 
        const char* spec = specs[i].c_str(); 
        bool exists= spath::Exists(spec); 
        std::cout 
            << " spec " << std::setw(100) << spec 
            << " exists " << ( exists ? "YES" : "NO " )
            << std::endl 
            ;
    }
}

void test_Exists2()
{
    const char* base = "$OPTICKS_HOME/sysrap/tests" ; 

    std::vector<std::string> names = {"hello.cc", "spath_test.cc", "spath_test.sh"} ; 
    for(unsigned i=0 ; i < names.size() ; i++)
    { 
        const char* name = names[i].c_str(); 
        const char* path = spath::Resolve(base, name); 
        bool exists= spath::Exists(base, name); 
        std::cout 
            << " path " << std::setw(100) << path 
            << " exists " << ( exists ? "YES" : "NO " )
            << std::endl 
            ;
    }
}


void test_Basename()
{
    const char* path = "/tmp/some/long/path/with/an/intersesting/base" ; 
    const char* base = spath::Basename(path) ; 
    assert( strcmp( base, "base") == 0 ); 
}


int main(int argc, char** argv)
{
    /*
    test_ResolvePath(); 
    test_Resolve(); 
    test_Exists(); 
    test_Exists2(); 
    */
    test_Basename(); 


    return 0 ; 
}

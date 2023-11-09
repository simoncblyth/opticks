// ./spath_test.sh

#include <cassert>
#include <iostream>
#include <iomanip>

#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"


struct spath_test 
{
   static void Resolve_with_undefined_token();
   static void Resolve_with_undefined_TMP();
   static void IsTokenWithFallback(); 
   static void ResolveTokenWithFallback(); 
   static void ResolvePath(); 
   static void Resolve(); 
   static void Exists(); 
   static void Exists2(); 
   static void Basename(); 
   static void Name(); 
   static void Remove(); 
};


void spath_test::Resolve_with_undefined_token()
{
   const char* path_ = "$HOME/.opticks/GEOM/$TYPO/CSGFoundry" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
}

void spath_test::Resolve_with_undefined_TMP()
{
   const char* path_ = "$TMP/GEOM/$TYPO/CSGFoundry" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
}

void spath_test::IsTokenWithFallback()
{
    const char* token = "{U4Debug_SaveDir:-$TMP}" ; 
    bool is_twf = spath::IsTokenWithFallback(token) ; 

    std::cout << " token " << token << " spath::IsTokenWithFallback " << ( is_twf ? "YES" : "NO " ) << std::endl ; 
}

void spath_test::ResolveTokenWithFallback()
{
    const char* token = "{U4Debug_SaveDir:-$TMP}" ; 
    const char* val = spath::_ResolveTokenWithFallback(token) ; 

    std::cout 
       << "spath_test::ResolveTokenWithFallback"
       << std::endl 
       << " token " << token 
       << std::endl 
       << " val   " << ( val ? val : "-" )
       << std::endl 
       ;
}


void spath_test::ResolvePath()
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
        const char* path = spath::Resolve(spec); 
        std::cout 
            << " spec " << spec << std::endl  
            << " path " << path << std::endl
            << std::endl 
            ;
    }
}

void spath_test::Resolve()
{
    const char* path0 = spath::Resolve("$HOME", "subdir", "another", "name.npy" ) ;  
    const char* path1 = spath::Resolve("$HOME/subdir/another/name.npy" ) ;  

    std::cout << " path0 [" << path0 << "]" << std::endl ;   
    std::cout << " path1 [" << path1 << "]" << std::endl ;   
}

void spath_test::Exists()
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

void spath_test::Exists2()
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


void spath_test::Basename()
{
    const char* path = "/tmp/some/long/path/with/an/intersesting/base" ; 
    const char* base = spath::Basename(path) ; 
    assert( strcmp( base, "base") == 0 ); 
}

void spath_test::Name()
{
    const char* name = spath::Name("red_","green_","blue") ; 
    assert( strcmp( name, "red_green_blue") == 0 ); 
}


void spath_test::Remove()
{
    const char* path_ = "/tmp/$USER/opticks/spath_test/file_test_Remove.gdml" ; 
    const char* path = spath::Resolve(path_); 
    sdirectory::MakeDirsForFile(path,0); 
    sstr::Write(path, "42:test_Remove\n" ); 

    std::cout 
        << " path_ " << path_
        << " path  " << path
        << std::endl 
        ;

    bool exists_0 = spath::Exists(path);  
    int rc = spath::Remove(path);  
    bool exists_1 = spath::Exists(path);  

    std::cout
        << " path_ " << path_
        << " path " << path 
        << " rc " << rc 
        << " exists_0 " << exists_0 
        << " exists_1 " << exists_1 
        << std::endl 
        ; 

    assert( exists_0 == 1); 
    assert( exists_1 == 0); 
}


int main(int argc, char** argv)
{
    /*
    spath_test::Resolve_with_undefined_token();
    spath_test::Resolve_with_undefined_TMP();
    spath_test::ResolvePath(); 
    spath_test::Resolve(); 
    spath_test::Exists(); 
    spath_test::Exists2(); 
    spath_test::Basename(); 
    spath_test::Name(); 
    spath_test::Remove(); 
    */
    spath_test::IsTokenWithFallback(); 
    spath_test::ResolveTokenWithFallback(); 

    return 0 ; 
}

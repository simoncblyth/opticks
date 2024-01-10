// ~/opticks/sysrap/tests/spath_test.sh

#include <cassert>
#include <iostream>
#include <iomanip>

#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"


struct spath_test 
{
   static void Resolve_inline(); 
   static void Resolve_defaultOutputPath(); 
   static void Resolve_with_undefined_token();
   static void Resolve_with_undefined_TMP();
   static void IsTokenWithFallback(); 
   static void ResolveTokenWithFallback(); 
   static void _ResolveToken(); 
   static void ResolveToken(); 
   static void ResolveToken_(const char* token); 
   static void ResolveToken1(); 

   static void Resolve_(const char* spec); 
   static void Resolve1(); 
   static void Resolve(); 

   static void Exists(); 
   static void Exists2(); 
   static void Basename(); 
   static void Name(); 
   static void Remove(); 

   static void _Check(); 
   static void Write(); 

   static int Main();  
};







void spath_test::Resolve_inline()
{
   const char* path_ = "$TMP/$ExecutableName/ALL${VERSION:-0}" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
}

void spath_test::Resolve_defaultOutputPath()
{
   const char* path_ = "$TMP/GEOM/$GEOM/$ExecutableName" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
}





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
    std::cout << "\nspath_test::ResolveTokenWithFallback\n\n" ;  

    std::vector<std::string> tokens = {
        "{U4Debug_SaveDir:-$TMP}",
        "${U4Debug_SaveDir:-$TMP}",
        "{VERSION:-99}",
        "${VERSION:-99}"
        }; 

    for(unsigned i=0 ; i < tokens.size() ; i++)
    {
        const char* token = tokens[i].c_str(); 
        const char* result = spath::_ResolveTokenWithFallback(token); 
        std::cout 
            << " token " << token << std::endl  
            << " result " << ( result ? result : "-" ) << std::endl
            << std::endl 
            ;
    }
}

void spath_test::_ResolveToken()
{
    std::cout << "\nspath_test::_ResolveToken\n\n" ;  
    std::vector<std::string> tokens = {
        "$TMP",
        "TMP" ,
        "$ExecutableName",
        "ExecutableName",
        }; 

    for(unsigned i=0 ; i < tokens.size() ; i++)
    { 
        const char* token = tokens[i].c_str(); 
        const char* result = spath::ResolveToken(token); 
        std::cout 
            << " token " << token << std::endl  
            << " result " << ( result ? result : "-" ) << std::endl
            << std::endl 
            ;
    }
}


void spath_test::ResolveToken_(const char* token)
{
    const char* result = spath::ResolveToken(token); 
    std::cout 
        << "spath_test::ResolveToken_" << std::endl 
        << " token " << token << std::endl  
        << " result " << ( result ? result : "-" ) << std::endl
        << std::endl 
        ;
}

void spath_test::ResolveToken1()
{
    //ResolveToken_("$DefaultOutputDir"); 
    ResolveToken_("{RNGDir:-$HOME/.opticks/rngcache/RNG}") ;
}
void spath_test::ResolveToken()
{
    std::cout << "\nspath_test::ResolveToken\n\n" ;  
    std::vector<std::string> tokens = {
        "$TMP",
        "${TMP:-/some/other/path}",
        "TMP" ,
        "${VERSION:-0}"
        }; 

    for(unsigned i=0 ; i < tokens.size() ; i++)
    { 
        const char* token = tokens[i].c_str(); 
        ResolveToken_(token) ; 
    }
}



void spath_test::Resolve_(const char* spec)
{
    //const char* path = spath::Resolve(spec); 
    const char* path = spath::ResolvePathGeneralized(spec); 
    std::cout 
        << " spec " << spec << std::endl  
        << " path " << path << std::endl
        << std::endl 
        ;
}

void spath_test::Resolve1()
{
    //Resolve_("${RNGDir:-$HOME/.opticks/rngcache/RNG}") ;
    Resolve_("$DefaultOutputDir") ;
}


void spath_test::Resolve()
{
    std::cout << "\nspath_test::Resolve\n\n" ;  
    std::vector<std::string> specs = {
        "$HOME/hello.npy", 
        "$HOME", 
        "$HOME/red/green/blue$HOME",
        "$HOME/red/green/blue$HOME/cyan/magenta",
        "${VERSION:-99}",
        "ALL${VERSION:-99}",
        "$TMP/GEOM/$GEOM/$ExecutableName/ALL${VERSION:-0}",
        "$TMP/GEOM/$GEOM/$ExecutableName/ALL${VERSION:-0}/tail",
        "$DefaultOutputDir",
        "$DefaultOutputDir/some/further/relative/path",
        "${RNGDir:-$HOME/.opticks/rngcache/RNG}",
        "${SEvt__INPUT_PHOTON_DIR:-$HOME/.opticks/InputPhotons}"
        } ; 

    for(unsigned i=0 ; i < specs.size() ; i++) Resolve_( specs[i].c_str() ); 
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

void spath_test::_Check()
{
    std::string chk0 = spath::_Check('A', "red", "green", "blue"); 
    std::string chk1 = spath::_Check('B', "red", "green", "blue"); 
    std::cout << "spath_test::_Check [" << chk0 << "]" << std::endl ; 
    std::cout << "spath_test::_Check [" << chk1 << "]" << std::endl ; 
}


void spath_test::Write()
{
    const char* path = "$TMP/spath_test_Write/some/deep/path/spath_test.txt" ; 
    bool ok0 = spath::Write("hello\n", path ); 
    std::cout << __FUNCTION__ << std::endl << path << " ok0 " << ( ok0 ? "YES" : "NO " ) << std::endl; 

    const char* base = "$TMP/spath_test_Write/some/deep/path" ;
    const char* name = "spath_test.txt" ; 
    bool ok1 = spath::Write("world\n", base, name ); 
    std::cout 
        << __FUNCTION__  << std::endl 
        << base << std::endl 
        << name << std::endl 
        << " ok1 " << ( ok1 ? "YES" : "NO " ) 
        << std::endl
        ; 

   /*
    // comment as writes into invoking directory 

    bool ok2 = spath::Write("jelly\n", nullptr, name ); 
    std::cout 
        << __FUNCTION__  << std::endl 
        << " base " << ( base ? base : "-" )  << std::endl 
        << " name " << ( name ? name : "-" ) << std::endl 
        << " ok2 " << ( ok2 ? "YES" : "NO " ) 
        << std::endl
        ; 
   */


}

int spath_test::Main()
{
/*
    Resolve_defaultOutputPath();
    Resolve_with_undefined_token();
    Resolve_with_undefined_TMP();
    Resolve_inline();
    ResolveToken(); 
    Resolve(); 
    Exists(); 
    Exists2(); 
    Basename(); 
    Name(); 
    Remove(); 
    IsTokenWithFallback(); 
    ResolveTokenWithFallback(); 

    ResolveTokenWithFallback(); 
    _ResolveToken(); 
    Resolve(); 
    ResolveToken1(); 
    Resolve1(); 
    _Check(); 
    Write(); 
*/
    ResolveToken1(); 

    return 0 ; 
}

 

int main(int argc, char** argv)
{
    return spath_test::Main(); 
}


// ~/opticks/sysrap/tests/spath_test.sh



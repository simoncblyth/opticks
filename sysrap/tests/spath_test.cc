/**
spath_test.cc
===============

::

     ~/opticks/sysrap/tests/spath_test.sh
     TEST=ResolveToken ~/opticks/sysrap/tests/spath_test.sh
     TEST=Resolve      ~/opticks/sysrap/tests/spath_test.sh
     TEST=Resolve3     ~/opticks/sysrap/tests/spath_test.sh
     TEST=Resolve_setenvvar ~/opticks/sysrap/tests/spath_test.sh
     TEST=Resolve_setenvmap ~/opticks/sysrap/tests/spath_test.sh
       
     TEST=DefaultOutputPath ~/opticks/sysrap/tests/spath_test.sh



**/

#include <cassert>
#include <iostream>
#include <iomanip>

#include "ssys.h"
#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"


struct spath_test 
{
   static int Resolve_inline(); 
   static int Resolve_defaultOutputPath(); 
   static int DefaultOutputPath();  

   static int Resolve_with_undefined_token();
   static int Resolve_with_undefined_TMP();
   static int IsTokenWithFallback(); 
   static int ResolveTokenWithFallback(); 
   static int _ResolveToken(); 
   static int ResolveToken(); 
   static int ResolveToken_(const char* token); 
   static int ResolveToken1(); 

   static int Resolve_(const char* spec); 
   static int Resolve1(); 
   static int Resolve3(); 
   static int Resolve(); 
   static int Resolve_setenvvar(); 
   static int Resolve_setenvmap(); 

   static int Exists(); 
   static int Exists2(); 
   static int Basename(); 
   static int Name(); 
   static int Remove(); 

   static int _Check(); 
   static int Write(); 
   static int WriteIntoInvokingDirectory();
   static int Read(); 
   static int EndsWith(); 
   static int SplitExt(int impl); 
   static int Filesize(); 

   static int ALL();  
   static int Main();  
};



int spath_test::Resolve_inline()
{
    const char* path_ = "$TMP/$ExecutableName/ALL${VERSION:-0}" ; 
    const char* path = spath::Resolve(path_); 
    std::cout 
        << " path_ [" << path_ << "]" << std::endl 
        << " path  [" << path  << "]" << std::endl 
        ;
    return 0 ; 
}

int spath_test::Resolve_defaultOutputPath()
{
    const char* path_ = "$TMP/GEOM/$GEOM/$ExecutableName" ; 
    const char* path = spath::Resolve(path_); 
    std::cout 
        << " path_ [" << path_ << "]" << std::endl 
        << " path  [" << path  << "]" << std::endl 
        ;
    return 0 ; 
}



int spath_test::DefaultOutputPath()
{
    //const char* stem = "stem_" ; 
    //const char* stem = "%Y%m%d_" ; 
    //const char* stem = "%Y%m%d_%H%M%S_" ; 
    const char* stem = nullptr ; 
    int index = 0 ; 
    const char* ext = ".txt" ; 
    bool unique = true ; 

    const char* path = spath::DefaultOutputPath(stem, index, ext, unique); 

    spath::MakeDirsForFile(path); 

    const char* txt = "yo : hello from spath_test::DefaultOutputPath\n" ; 
    sstr::Write(path, txt); 

    std::cout 
        << __FUNCTION__  << std::endl 
        << " path [" << ( path ? path : "-" ) 
        << std::endl 
        ; 

    return 0 ; 
}





int spath_test::Resolve_with_undefined_token()
{
   const char* path_ = "$HOME/.opticks/GEOM/$TYPO/CSGFoundry" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
    return 0 ; 
}

int spath_test::Resolve_with_undefined_TMP()
{
   const char* path_ = "$TMP/GEOM/$TYPO/CSGFoundry" ; 
   const char* path = spath::Resolve(path_); 
   std::cout 
       << " path_ [" << path_ << "]" << std::endl 
       << " path  [" << path  << "]" << std::endl 
       ;
    return 0 ; 
}

int spath_test::IsTokenWithFallback()
{
    const char* token = "{U4Debug_SaveDir:-$TMP}" ; 
    bool is_twf = spath::IsTokenWithFallback(token) ; 

    std::cout << " token " << token << " spath::IsTokenWithFallback " << ( is_twf ? "YES" : "NO " ) << std::endl ; 
    return 0 ; 
}

int spath_test::ResolveTokenWithFallback()
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
    return 0 ; 
}

int spath_test::_ResolveToken()
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
    return 0 ; 
}


int spath_test::ResolveToken_(const char* token)
{
    const char* result = spath::ResolveToken(token); 
    std::cout 
        << "spath_test::ResolveToken_" << std::endl 
        << " token " << token << std::endl  
        << " result " << ( result ? result : "-" ) << std::endl
        << std::endl 
        ;
    return 0 ; 
}

int spath_test::ResolveToken1()
{
    //ResolveToken_("$DefaultOutputDir"); 
    ResolveToken_("{RNGDir:-$HOME/.opticks/rngcache/RNG}") ;
    return 0 ; 
}
int spath_test::ResolveToken()
{
    std::cout << "\nspath_test::ResolveToken\n\n" ;  
    std::vector<std::string> tokens = {
        "$TMP",
        "${TMP}",
        "TMP" ,
        "${TMP:-/some/other/path}",
        "${VERSION:-0}"
        }; 

    for(unsigned i=0 ; i < tokens.size() ; i++)
    { 
        const char* token = tokens[i].c_str(); 
        ResolveToken_(token) ; 
    }
    return 0 ; 
}



int spath_test::Resolve_(const char* spec)
{
    //const char* path = spath::Resolve(spec); 
    const char* path = spath::ResolvePathGeneralized(spec); 
    std::cout 
        << " spec " << spec << std::endl  
        << " path " << path << std::endl
        << std::endl 
        ;
    return 0 ; 
}

int spath_test::Resolve1()
{
    std::vector<std::string> pp = { 
       "${RNGDir:-$HOME/.opticks/rngcache/RNG}", 
       "$DefaultOutputDir",
       "$DefaultOutputDir/$TEST",
       "$DefaultOutputDir/${TEST:-notest}",
     } ; 

    for(int i=0 ; i < int(pp.size()) ; i++)
    {
        const char* p = pp[i].c_str(); 
        const char* path0 = spath::ResolvePath(p) ;
        const char* path1 = spath::ResolvePathGeneralized(p) ;
        std::cout 
            << "spath_test::Resolve1\n"
            << " p                             : " << p << "\n"
            << " spath::ResolvePath            : " << ( path0 ? path0 : "-" ) << "\n"
            << " spath::ResolvePathGeneralized : " << ( path1 ? path1 : "-" ) << "\n"
            << "\n"
            ;
    }

    return 0 ; 
}

int spath_test::Resolve3()
{
    const char* base = "$TMP/GEOM/$GEOM/$ExecutableName" ; 
    const char* sidx = "A000" ; 

    std::vector<std::string> pp = { 
         "ALL${VERSION:-0}", 
         "ALL${VERSION:-0}${TEST:-}", 
         "ALL${VERSION:-0}${XTEST:-}",
         "ALL${VERSION:-0}${XTEST:-notest}"
     } ; 
 
 
    for(int i=0 ; i < int(pp.size()) ; i++)
    {
        const char* reldir = pp[i].c_str(); 
        const char* path = spath::Resolve(base,reldir,sidx ) ; 
        std::cout 
            << "spath_test::Resolve3\n"
            << " base   : " << base << "\n"
            << " reldir : " << reldir << "\n"
            << " sidx   : " << sidx << "\n"
            << " path   : " << ( path ? path : "-" ) << "\n\n"
            ;
    }

    return 0 ; 

}

int spath_test::Resolve()
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
        "$TMP",
        "${TMP}",
        "${TMP:-/some/fallback}",
        "${XMP:-/some/fallback}",
        "TMP",
        "{TMP}",
        "$DefaultOutputDir",
        "$DefaultOutputDir/some/further/relative/path",
        "${RNGDir:-$HOME/.opticks/rngcache/RNG}",
        "${SEvt__INPUT_PHOTON_DIR:-$HOME/.opticks/InputPhotons}",
        "${PrecookedDir:-$HOME/.opticks/precooked}/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy"
        } ; 

    for(unsigned i=0 ; i < specs.size() ; i++) Resolve_( specs[i].c_str() ); 
    return 0 ; 
}

int spath_test::Resolve_setenvvar()
{
    std::cout << "\nspath_test::Resolve_setenvvar\n\n" ;  
    std::vector<std::string> eye = {
        "0,0,0", 
        "1,1,1", 
        "2,2,2" 
        } ; 

    std::vector<std::string> look = {
        "0,0,0", 
        "0,0,0", 
        "10,10,10" 
        } ; 

    std::vector<std::string> up = {
        "0,0,1", 
        "0,0,2", 
        "0,0,3" 
        } ; 


    assert( eye.size() == look.size() ); 
    assert( eye.size() == up.size() ); 

    const char* spec = "EYE=${EYE}_LOOK=${LOOK}_UP=${UP}" ; 
    for(unsigned i=0 ; i < eye.size() ; i++)
    {
         ssys::setenvctx( 
                 "EYE", eye[i].c_str(),
                 "LOOK", look[i].c_str(),
                 "UP", up[i].c_str() ); 

         Resolve_( spec );
    } 
    return 0 ; 
}



int spath_test::Resolve_setenvmap()
{
    std::cout << "\nspath_test::Resolve_setenvmap\n\n" ;  

    std::map<std::string, std::string> m = 
       {
          { "EYE", "1,1,0" },
          { "LOOK", "0,0,0" },
          { "UP", "0,0,1" },
        };

    const char* spec = "EYE=${EYE}_LOOK=${LOOK}_UP=${UP}" ; 
    ssys::setenvmap( m ); 
    Resolve_( spec );

    return 0 ; 
}



int spath_test::Exists()
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
    return 0 ; 
}

int spath_test::Exists2()
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
    return 0 ; 
}


int spath_test::Basename()
{
    const char* path = "/tmp/some/long/path/with/an/intersesting/base" ; 
    const char* base = spath::Basename(path) ; 
    assert( strcmp( base, "base") == 0 ); 
    return 0 ; 
}

int spath_test::Name()
{
    const char* name = spath::Name("red_","green_","blue") ; 
    assert( strcmp( name, "red_green_blue") == 0 ); 
    return 0 ; 
}


int spath_test::Remove()
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
    return 0 ; 
}

int spath_test::_Check()
{
    std::string chk0 = spath::_Check('A', "red", "green", "blue"); 
    std::string chk1 = spath::_Check('B', "red", "green", "blue"); 
    std::cout << "spath_test::_Check [" << chk0 << "]" << std::endl ; 
    std::cout << "spath_test::_Check [" << chk1 << "]" << std::endl ; 
    return 0 ; 
}


int spath_test::Write()
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

    return 0 ; 
}

int spath_test::WriteIntoInvokingDirectory()
{
    const char* name = "spath_test.txt" ; 
    bool ok2 = spath::Write("jelly\n", nullptr, name ); 
    std::cout 
        << __FUNCTION__  << std::endl 
        << " name " << ( name ? name : "-" ) << std::endl 
        << " ok2 " << ( ok2 ? "YES" : "NO " ) 
        << std::endl
        ; 
    return 0 ; 
}

int spath_test::Read()
{
    std::vector<char> data ; 
    bool ok = spath::Read(data, "$EXECUTABLE" ); 
    std::cout 
        << __FUNCTION__  << std::endl 
        << " ok " << ( ok ? "YES" : "NO " ) 
        << " data.size " << data.size()
        << std::endl
        ; 
    return 0 ; 
}

int spath_test::EndsWith()
{
    bool ok = spath::EndsWith("$EXECUTABLE", "spath_test" ); 
    std::cout 
        << __FUNCTION__  << std::endl 
        << " ok " << ( ok ? "YES" : "NO " ) 
        << std::endl
        ; 
    return ok ? 0 : 1  ; 
}

int spath_test::SplitExt(int impl)
{
    const char* path = "/tmp/blyth/opticks/GEOM/J_2024aug27/CSGOptiXRenderTest/CVD1/70500/ALL/scan-emm/cxr_overview_emm_t10,_elv_t_moi__ALL00000.jpg" ; 

    std::string dir ; 
    std::string stem ; 
    std::string ext ; 

    int rc = impl == 0 ? spath::SplitExt0(dir, stem, ext, path ) : spath::SplitExt(dir, stem, ext, path ) ; 

    std::cout 
        << __FUNCTION__  << " impl " << impl << "\n"
        << " path [" << path << "]\n" 
        << " dir  [" << dir  << "]\n"
        << " stem [" << stem << "]\n"
        << " ext  [" << ext  << "]\n" 
        << " rc   [" << rc << "]\n"
        << std::endl
        ; 
    return rc  ; 
}

int spath_test::Filesize()
{
    long sz = spath::Filesize(spath::CWD(), __FILE__); 

    std::cout 
        << "spath_test::Filesize"
        << "[" << __FILE__ << "]"
        << " sz : " << sz
        << "\n"
        ;
  
    return sz > 0 ;
}


int spath_test::ALL()
{
    int rc = 0 ; 

    rc += Resolve_defaultOutputPath() ; 
    //rc += DefaultOutputPath();      // comment as writes
    rc += Resolve_with_undefined_token();
    rc += Resolve_with_undefined_TMP();
    rc += Resolve_inline();
    rc += ResolveToken(); 
    rc += Resolve(); 
    rc += Exists(); 
    rc += Exists2(); 
    rc += Basename(); 
    rc += Name(); 
    rc += Remove(); 
    rc += IsTokenWithFallback(); 
    rc += ResolveTokenWithFallback(); 
    rc += _ResolveToken(); 
    rc += Resolve(); 
    rc += Resolve_setenvvar(); 
    rc += Resolve_setenvmap(); 
    rc += ResolveToken1(); 
    rc += Resolve1(); 
    rc += _Check(); 
    rc += Write(); 
    //rc += WriteIntoInvokingDirectory();   // comment as leaves droppings
    //rc += Read(); 

    return rc ; 
}


int spath_test::Main()
{
    //const char* test = "Resolve_defaultOutputPath" ;
    //const char* test = "Read" ;
    //const char* test = "EndsWith" ;
    const char* test = "SplitExt" ;
 
    const char* TEST = ssys::getenvvar("TEST", test ); 
    int rc = 0 ; 
    if(     strcmp(TEST, "Resolve_defaultOutputPath")==0 )   rc = Resolve_defaultOutputPath();
    else if(strcmp(TEST, "DefaultOutputPath")==0) rc = DefaultOutputPath();
    else if(strcmp(TEST, "Resolve_with_undefined_token")==0) rc = Resolve_with_undefined_token();
    else if(strcmp(TEST, "Resolve_with_undefined_TMP")==0) rc = Resolve_with_undefined_TMP();
    else if(strcmp(TEST, "Resolve_inline")==0) rc = Resolve_inline();
    else if(strcmp(TEST, "ResolveToken")==0) rc = ResolveToken();
    else if(strcmp(TEST, "Resolve")==0) rc = Resolve();
    else if(strcmp(TEST, "Exists")==0) rc = Exists();
    else if(strcmp(TEST, "Exists2")==0) rc = Exists2();
    else if(strcmp(TEST, "Basename")==0) Basename();
    else if(strcmp(TEST, "Name")==0) rc = Name();
    else if(strcmp(TEST, "Remove")==0) rc = Remove();
    else if(strcmp(TEST, "IsTokenWithFallback")==0) rc = IsTokenWithFallback();
    else if(strcmp(TEST, "ResolveTokenWithFallback")==0) rc = ResolveTokenWithFallback();
    else if(strcmp(TEST, "_ResolveToken")==0) rc = _ResolveToken();
    else if(strcmp(TEST, "Resolve")==0) rc = Resolve();
    else if(strcmp(TEST, "Resolve_setenvvar")==0) rc = Resolve_setenvvar();
    else if(strcmp(TEST, "Resolve_setenvmap")==0) rc = Resolve_setenvmap();
    else if(strcmp(TEST, "ResolveToken1")==0) rc = ResolveToken1();
    else if(strcmp(TEST, "Resolve1")==0) rc = Resolve1();
    else if(strcmp(TEST, "Resolve3")==0) rc = Resolve3();
    else if(strcmp(TEST, "_Check")==0) rc = _Check();
    else if(strcmp(TEST, "Write")==0) rc = Write();
    else if(strcmp(TEST, "WriteIntoInvokingDirectory")==0) rc = WriteIntoInvokingDirectory();
    else if(strcmp(TEST, "Read")==0) rc = Read();
    else if(strcmp(TEST, "EndsWith")==0) rc = EndsWith();
    else if(strcmp(TEST, "SplitExt0")==0) rc = SplitExt(0);
    else if(strcmp(TEST, "SplitExt")==0) rc = SplitExt(1);
    else if(strcmp(TEST, "Filesize")==0) rc = Filesize();
    else if(strcmp(TEST, "ALL")==0) rc = ALL();
    return rc ; 
}

int main(int argc, char** argv)
{
    return spath_test::Main(); 
}



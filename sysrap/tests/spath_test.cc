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

     TEST=CFBaseFromGEOM ~/opticks/sysrap/tests/spath_test.sh
     TEST=is_readable ~/opticks/sysrap/tests/spath_test.sh
     TEST=GEOMSub ~/opticks/sysrap/tests/spath_test.sh
     TEST=GDMLPathFromGEOM ~/opticks/sysrap/tests/spath_test.sh
     TEST=Dirname ~/opticks/sysrap/tests/spath_test.sh
     TEST=Dirname0 ~/opticks/sysrap/tests/spath_test.sh

**/

#include <cassert>
#include <iostream>
#include <iomanip>


#include "ssys.h"
#include "sstr.h"
#include "spath.h"
#include "sdirectory.h"
#include "sstamp.h"


struct spath_test
{
   static int Resolve_name();
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
   static int LooksUnresolved_0();
   static int LooksUnresolved_1();
   static int Resolve_setenvvar();
   static int Resolve_setenvmap();

   static int Exists();
   static int Exists2();
   static int Dirname0();
   static int Dirname();
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
   static int CFBaseFromGEOM();
   static int is_readable();
   static int GEOMSub();
   static int GDMLPathFromGEOM();
   static int last_write_time();

   static int Main();
};

int spath_test::Resolve_name()
{
    const char* name_ = "${SGLFW_Evt__shader_name:-rec_flying_point_persist}" ;
    const char* name = spath::Resolve(name_);
    std::cout
        << " name_ [" << name_ << "]" << std::endl
        << " name  [" << name  << "]" << std::endl
        ;
    return 0 ;
}


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
    std::cout << "[spath_test::Resolve_defaultOutputPath\n" ;
    const char* path_ = "$TMP/GEOM/$GEOM/$ExecutableName" ;
    const char* path = spath::Resolve(path_);
    std::cout
        << " path_ [" << path_ << "]" << std::endl
        << " path  [" << path  << "]" << std::endl
        ;
    std::cout << "]spath_test::Resolve_defaultOutputPath\n" ;
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
   std::cout << "[spath_test::Resolve_with_undefined_token\n" ;

   const char* path_ = "$HOME/.opticks/GEOM/$TYPO/CSGFoundry" ;
   const char* path = spath::Resolve(path_);
   std::cout
       << " path_ [" << path_ << "]" << std::endl
       << " path  [" << path  << "]" << std::endl
       ;
   std::cout << "]spath_test::Resolve_with_undefined_token\n" ;
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
        "${PrecookedDir:-$HOME/.opticks/precooked}/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy",
        "${CSGOptiX__cu_ptx:-$OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx}"
        } ;

    for(unsigned i=0 ; i < specs.size() ; i++) Resolve_( specs[i].c_str() );
    return 0 ;
}

int spath_test::LooksUnresolved_0()
{
    const char* _path = "$SOME_NON_EXISTING_TOKEN/And/Something/More" ;
    const char* path = spath::Resolve(_path);
    std::cout << "LooksUnresolved_0[" << ( path ? path : "-" ) << "]\n" ;

    const char* xpath = "SOME_NON_EXISTING_TOKEN/And/Something/More" ;
    assert( strcmp( path, xpath) == 0 );

    bool unresolved = spath::LooksUnresolved(path, _path);
    assert( unresolved == true );
    return 0 ;
}

int spath_test::LooksUnresolved_1()
{
    const char* _path = "$SOME_NON_EXISTING_TOKEN/And/Something/More" ;
    const char* path = spath::Resolve(_path, "record.npy");
    std::cout << "LooksUnresolved_1[" << ( path ? path : "-" ) << "]\n" ;

    const char* xpath = "SOME_NON_EXISTING_TOKEN/And/Something/More/record.npy" ;
    assert( strcmp( path, xpath) == 0 );

    bool unresolved = spath::LooksUnresolved(path, _path);
    assert( unresolved == true );
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



int spath_test::Dirname0()
{
    const char* path = "record.npy" ;
    const char* fold = spath::Dirname(path) ;
    bool expect = strcmp( fold, "") == 0 ;

    std::cout
        << "spath_test::Dirname0\n"
        << " path [" << path << "]\n"
        << " fold [" << ( fold ? fold : "-" ) << "]\n"
        << " expect [" << (expect ? "YES" : "NO " ) << "]\n"
        ;

    assert( expect );
    return 0 ;
}



int spath_test::Dirname()
{
    const char* path = "/tmp/some/long/path/with/an/interesting/base" ;
    const char* fold = spath::Dirname(path) ;
    bool expect = strcmp( fold, "/tmp/some/long/path/with/an/interesting") == 0 ;

    std::cout
        << "spath_test::Dirname\n"
        << " path [" << path << "]\n"
        << " fold [" << ( fold ? fold : "-" ) << "]\n"
        << " expect [" << (expect ? "YES" : "NO " ) << "]\n"
        ;

    assert( expect );
    return 0 ;
}

int spath_test::Basename()
{
    const char* path = "/tmp/some/long/path/with/an/interesting/base" ;
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

    return sz > 0 ? 0 : 1 ;
}

int spath_test::CFBaseFromGEOM()
{
    const char* cfb = spath::CFBaseFromGEOM() ;
    const char* ori = spath::Resolve("$CFBaseFromGEOM/origin.gdml") ;

    bool has = spath::has_CFBaseFromGEOM() ;

    std::cout
         << " spath::CFBaseFromGEOM()                         [" << ( cfb ? cfb : "-" ) << "]\n"
         << " spath::Resolve(\"$CFBaseFromGEOM/origin.gdml\")   [" << ( ori ? ori : "-" ) << "]\n"
         << " spath::has_CFBaseFromGEOM()                     [" << ( has ? "YES" : "NO " ) << "]\n"
         ;
    return 0 ;
}

int spath_test::is_readable()
{
    bool is0 = spath::is_readable("$CFBaseFromGEOM/origin.gdml") ;
    bool is1 = spath::is_readable("$CFBaseFromGEOM", "origin.gdml") ;
    bool is2 = spath::is_readable("$CFBaseFromGEOM") ;
    bool is3 = spath::is_readable("$CFBaseFromGEOM/") ;
    bool is4 = spath::is_readable("$SomeNonExistingToken") ;
    bool is5 = spath::is_readable("$CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt") ;


    std::cout
         << " spath::is_readable(\"$CFBaseFromGEOM/origin.gdml\")    [" << ( is0 ? "YES" : "NO " ) << "]\n"
         << " spath::is_readable(\"$CFBaseFromGEOM\",\"origin.gdml\")  [" << ( is1 ? "YES" : "NO " ) << "]\n"
         << " spath::is_readable(\"$CFBaseFromGEOM\")                [" << ( is2 ? "YES" : "NO " ) << "]\n"
         << " spath::is_readable(\"$CFBaseFromGEOM/\")               [" << ( is3 ? "YES" : "NO " ) << "]\n"
         << " spath::is_readable(\"$SomeNonExistingToken\")          [" << ( is4 ? "YES" : "NO " ) << "]\n"
         << " spath::is_readable(\"$CFBaseFromGEOM/CSGFoundry/SSim/extra/jpmt\")          [" << ( is5 ? "YES" : "NO " ) << "]\n"
         ;
    assert( is0 == is1 );
    return 0 ;
}

int spath_test::GEOMSub()
{
    const char* x_gsub = getenv("hello_GEOMSub");
    const char* gsub = spath::GEOMSub("hello") ;
    bool validmatch = x_gsub != nullptr && gsub != nullptr && strcmp( x_gsub, gsub ) == 0 ;

    std::cout
       << "spath_test::GEOMSub"
       << " x_gsub " << ( x_gsub ? x_gsub : "-" )
       << " gsub  " << ( gsub ? gsub : "-" )
       << " validmatch " << ( validmatch ? "YES" : "NO " )
       << "\n"
       ;

    //assert( validmatch );
    return 0 ;
}

int spath_test::GDMLPathFromGEOM()
{
    const char* path = spath::GDMLPathFromGEOM();

    std::cout
       << "spath_test::GDMLPathFromGEOM"
       << " [" << ( path ? path : "-" ) << "]\n"
       ;

    return 0 ;
}



int spath_test::last_write_time()
{
    const char* path = spath::Resolve("$HOME/.bash_profile");
    if(!spath::Exists(path)) return 0 ;

    int64_t now = sstamp::Now();
    int64_t mtime = spath::last_write_time(path);

    int64_t age_secs = sstamp::age_seconds(mtime);
    int64_t age_days = sstamp::age_days(mtime);

    std::cout
        << "spath_test::last_write_time\n"
        << " path [" << ( path ? path : "-" ) << "]\n"
        << " int64_t mtime = spath::last_write_time(\"" << path << "\")\n"
        << "        mtime  [" << mtime << "]\n"
        << " sstamp::Now() [" << now << "]\n"
        << " sstamp::Format(now)   [" << sstamp::Format(now) << "]\n"
        << " sstamp::Format(mtime) [" << sstamp::Format(mtime) << "]\n"
        << " age_secs " << age_secs << "\n"
        << " age_days " << age_days << "\n"
        ;

    return 0 ;
}

int spath_test::Main()
{
    const char* test = "ALL" ;

    const char* TEST = ssys::getenvvar("TEST", test );
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    std::cout << "[spath_test::Main TEST [" << ( TEST ? TEST : "-" ) << "]\n" ;

    int rc = 0 ;
    if(ALL||strcmp(TEST, "Resolve_defaultOutputPath")==0 )   rc += Resolve_defaultOutputPath();
    if(ALL||strcmp(TEST, "Resolve_with_undefined_token")==0) rc += Resolve_with_undefined_token();
    if(ALL||strcmp(TEST, "Resolve_with_undefined_TMP")==0) rc += Resolve_with_undefined_TMP();
    if(ALL||strcmp(TEST, "Resolve_name")==0) rc += Resolve_name();
    if(ALL||strcmp(TEST, "Resolve_inline")==0) rc += Resolve_inline();
    if(ALL||strcmp(TEST, "ResolveToken")==0) rc += ResolveToken();
    if(ALL||strcmp(TEST, "Resolve")==0) rc += Resolve();
    if(ALL||strcmp(TEST, "Exists")==0) rc += Exists();
    if(ALL||strcmp(TEST, "Exists2")==0) rc += Exists2();
    if(ALL||strcmp(TEST, "Dirname0")==0) Dirname0();
    if(ALL||strcmp(TEST, "Dirname")==0) Dirname();
    if(ALL||strcmp(TEST, "Basename")==0) Basename();
    if(ALL||strcmp(TEST, "Name")==0) rc += Name();
    if(ALL||strcmp(TEST, "Remove")==0) rc += Remove();
    if(ALL||strcmp(TEST, "IsTokenWithFallback")==0) rc += IsTokenWithFallback();
    if(ALL||strcmp(TEST, "ResolveTokenWithFallback")==0) rc += ResolveTokenWithFallback();
    if(ALL||strcmp(TEST, "_ResolveToken")==0) rc += _ResolveToken();
    if(ALL||strcmp(TEST, "Resolve")==0) rc += Resolve();
    if(ALL||strcmp(TEST, "LooksUnresolved_0")==0) rc += LooksUnresolved_0();
    if(ALL||strcmp(TEST, "LooksUnresolved_1")==0) rc += LooksUnresolved_1();
    if(ALL||strcmp(TEST, "Resolve_setenvvar")==0) rc += Resolve_setenvvar();
    if(ALL||strcmp(TEST, "Resolve_setenvmap")==0) rc += Resolve_setenvmap();
    if(ALL||strcmp(TEST, "ResolveToken1")==0) rc += ResolveToken1();
    if(ALL||strcmp(TEST, "Resolve1")==0) rc += Resolve1();
    if(ALL||strcmp(TEST, "Resolve3")==0) rc += Resolve3();
    if(ALL||strcmp(TEST, "_Check")==0) rc += _Check();
    if(ALL||strcmp(TEST, "Write")==0) rc += Write();
    if(ALL||strcmp(TEST, "EndsWith")==0) rc += EndsWith();
    if(ALL||strcmp(TEST, "SplitExt0")==0) rc += SplitExt(0);
    if(ALL||strcmp(TEST, "SplitExt")==0) rc += SplitExt(1);
    if(ALL||strcmp(TEST, "Filesize")==0) rc += Filesize();
    if(ALL||strcmp(TEST, "CFBaseFromGEOM")==0) rc += CFBaseFromGEOM();
    if(ALL||strcmp(TEST, "is_readable")==0) rc += is_readable();
    if(ALL||strcmp(TEST, "GEOMSub")==0) rc += GEOMSub();
    if(ALL||strcmp(TEST, "GDMLPathFromGEOM")==0) rc += GDMLPathFromGEOM();
    if(ALL||strcmp(TEST, "last_write_time")==0) rc += last_write_time();

    //if(ALL||strcmp(TEST, "DefaultOutputPath")==0) rc += DefaultOutputPath();
    //if(ALL||strcmp(TEST, "WriteIntoInvokingDirectory")==0) rc += WriteIntoInvokingDirectory();
    //if(ALL||strcmp(TEST, "Read")==0) rc += Read();

    std::cout << "]spath_test::Main TEST [" << ( TEST ? TEST : "-" ) << "]" << " rc " << rc << "\n" ;
    return rc ;
}

int main(int argc, char** argv)
{
    return spath_test::Main();
}



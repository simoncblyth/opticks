// ~/opticks/sysrap/tests/ssys_test.sh

#include <string>
#include <iostream>
#include <iomanip>
#include "ssys.h"


struct ssys_test
{
    static constexpr const char* getenviron_SIGINT = "ssys_test__getenviron_SIGINT" ;
    static int getenviron();
    static int getenvuint64spec();
};


/**
ssys_test::getenviron
-----------------------

Use this to check for envvars that identify running under ctest.

**/

int ssys_test::getenviron()
{
    bool _SIGINT = ssys::hasenv_(getenviron_SIGINT);
    std::cout
        << "[ssys_test::getenviron"
        << " [" << getenviron_SIGINT << "] " << ( _SIGINT ? "YES" : "NO " ) << "\n"
        << "-ssys_test::getenviron.CTEST\n"
        << ssys::getenviron("CTEST")
        << "-ssys_test::getenviron.DART\n"
        << ssys::getenviron("DART")
        << "-ssys_test::getenviron.ALL\n"
        << ssys::getenviron()
        << "]ssys_test::getenviron\n"
        ;

    if(_SIGINT) std::raise(SIGINT);
    return 0 ;
}


int ssys_test::getenvuint64spec()
{
    uint64_t pidx = ssys::getenvuint64spec("PIDX", "X40" );
    std::cout
         << "ssys_test::getenvuint64spec\n"
         << " pidx (hex) " << std::setw(15) << std::hex << pidx << std::dec
         << "\n"
         << " pidx (dec) " << std::setw(15) << pidx
         << "\n"
         ;

    return 0 ;
}





int test_popen_0()
{
    const char* cmd = "md5 -q -s hello" ;
    bool chomp = false ;
    std::string ret = ssys::popen(cmd, chomp);

    std::cout
        << " cmd [" << cmd << "]"
        << std::endl
        << " ret [" << ret << "]"
        << std::endl
        ;
    return 0 ;
}

int test_popen_1()
{
    std::cout << ssys::popen("md5 -q -s hello") << std::endl ;
    return 0 ;
}


int  test_is_remote_session()
{
    std::cout << "ssys::is_remote_session [" << ( ssys::is_remote_session() ? "YES" : "NO " ) << "]" << std::endl ;
    return 0 ;
}

int test_username()
{
    std::cout << "ssys::username [" << ssys::username() << "]" << std::endl ;
    return 0 ;
}
void test_which ()
{
    std::cout << "ssys::which(\"md5\") [" << ssys::which("md5") << "]" << std::endl ;
}



void test_getenv_()
{
    int i = ssys::getenv_<int>("i",0) ;
    unsigned u = ssys::getenv_<unsigned>("u",0) ;
    float f = ssys::getenv_<float>("f",0.f) ;
    double d = ssys::getenv_<double>("d",0.) ;
    std::string s = ssys::getenv_<std::string>("s","string") ;

    std::cout
        << "i " << i << std::endl
        << "u " << u << std::endl
        << "f " << f << std::endl
        << "d " << d << std::endl
        << "s " << s << std::endl
        ;
}

void test_getenv_vec()
{
    std::vector<int>* ivec = ssys::getenv_vec<int>("IVEC", "1,2,3,4" );
    assert( ivec->size() == 4 );
    std::cout << "IVEC " << ssys::desc_vec<int>(ivec) << std::endl ;
}

void test_getenvvec_string_1()
{
    std::vector<std::string>* svec = ssys::getenv_vec<std::string>("SVEC", "A1,B2,C3,D4" );
    assert( svec->size() == 4 );
    std::cout << "SVEC " << ssys::desc_vec<std::string>(svec) << std::endl ;
}



void test_getenvvec_string_2()
{
    std::vector<std::string>* svec = ssys::getenv_vec<std::string>("SVEC", "A1,B2,C3,D4" );
    std::cout << "SVEC " << ssys::desc_vec<std::string>(svec) << std::endl ;

    for(int i=0 ; i < int(svec->size()) ; i++)
    {
        const std::string& s = (*svec)[i] ;
        std::cout << "[" << s << "]" << std::endl ;
    }
}



void test_getenv_kv()
{
    typedef std::pair<std::string, std::string> KV ;
    std::vector<KV> kvs ;

    const char* kk = R"LITERAL(

    HOME
    CHECK
    OPTICKS_HOME
    COMMANDLINE
    ${GEOM}_GEOMList

)LITERAL" ;

    //std::cout << "test_getenv_kv" << kk << std::endl ;

    ssys::getenv_(kvs, kk);

    for(int i=0 ; i < int(kvs.size()); i++)
    {
        const KV& kv = kvs[i];
        std::cout << std::setw(20) << kv.first << " : " << kv.second << std::endl ;
    }
}

void test__getenv()
{
    const char* k = "${GEOM}_GEOMList" ;
    char* v = ssys::_getenv(k) ;

    std::cout
        << " k " << k
        << " v " << ( v ? v : "-" )
        << std::endl
        ;

}

void test_replace_envvar_token()
{
    const char* kk = R"LITERAL(
    HOME
    ${HOME}_HELLO
    WORLD_${HOME}_HELLO

    ALL${VERSION}HELLO
    ALL${VERSIONX}HELLO

)LITERAL" ;

    std::stringstream ss(kk) ;
    std::string str ;
    while (std::getline(ss, str))  // newlines are swallowed by getline
    {
        if(str.empty()) continue ;

        std::string rep = ssys::replace_envvar_token(str.c_str());

        std::cout
            << "[" << str << "]"
            << "[" << rep << "]"
            << std::endl
            ;
    }


}

void test_getenvfloat()
{
    float f = ssys::getenvfloat("f",0.314f) ;
    std::cout << "f:" << std::scientific << f << std::endl ;
}

void test_getenvdouble()
{
    double d = ssys::getenvdouble("d",1e-5 ) ;
    std::cout << "d:" << std::scientific << d << std::endl ;
}

void test_getenvvar()
{
    const char* ekey = "OPTICKS_ELV_SELECTION,ELV" ;
    const char* val = ssys::getenvvar(ekey) ;
    std::cout
        << "test_getenvvar"
        << " ekey " << ( ekey ? ekey : "-" )
        << " val " << ( val ? val : "-" )
        << std::endl
        ;
}

void test_getenvvar_path_replacement()
{
    const char* ekey = "stree__force_triangulate_solid" ;
    const char* _force_triangulate_solid = ssys::getenvvar(ekey, nullptr) ;
    char delim = ',' ;

    std::vector<std::string> force ;
    if( _force_triangulate_solid )
    {
        sstr::SplitTrimSuppress( _force_triangulate_solid, delim, force );
    }

    unsigned num_force = force.size();

    std::cout
        << "test_getenvvar_path_replacement "
        << " ekey " << ( ekey ? ekey : "-" )
        << " _force_triangulate_solid [" << ( _force_triangulate_solid ? _force_triangulate_solid : "-" ) << "]"
        << std::endl
        ;

    for(unsigned i=0 ; i < num_force ; i++)
    {
        const char* f = force[i].c_str() ;
        std::cout << std::setw(5) << i << ":[" << f << "]\n" ;
    }

}


void test_getenvfloatvec_fallback()
{
    const char* ekey = "UKEY" ;
    std::vector<float>* fvec = ssys::getenvfloatvec(ekey, "0,0,0");
    int num_values = fvec ? fvec->size() : -1 ;
    std::cout
         << __FUNCTION__
         << " ekey " << ekey
         << " num_values " << num_values
         << std::endl
         ;
}


void test_is_listed()
{
    std::vector<std::string> names = {"red", "green", "blue" } ;
    const std::vector<std::string>* nn = &names ;
    std::cout << __FUNCTION__ << " nn->size " << nn->size() << std::endl ;

    assert( ssys::is_listed(nn, "red")   == true );
    assert( ssys::is_listed(nn, "green") == true );
    assert( ssys::is_listed(nn, "blue")  == true );
    assert( ssys::is_listed(nn, "cyan")  == false );

    assert( ssys::is_listed(nullptr, "blue") == false );
}


void test_listed_count()
{

    std::vector<int> ncount ;
    std::vector<std::string> names = {"red", "green", "blue" } ;

    std::vector<int>* nc = &ncount ;
    const std::vector<std::string>* nn = &names ;

    assert( ssys::listed_count(nc, nn, "pink") == -1 ) ;
    assert( ssys::listed_count(nc, nn, "red") == 0 ) ;
    assert( ssys::listed_count(nc, nn, "red") == 1 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 0 ) ;
    assert( ssys::listed_count(nc, nn, "red") == 2 ) ;
    assert( ssys::listed_count(nullptr, nn, "red") == -1 ) ;
    assert( ssys::listed_count(nullptr, nullptr, "red") == -1 ) ;
    assert( ssys::listed_count(nc, nullptr, "red") == -1 ) ;
    assert( ssys::listed_count(nc, nn, "blue") == 0 ) ;
    assert( ssys::listed_count(nc, nn, "blue") == 1 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 1 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 2 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 3 ) ;
    assert( ssys::listed_count(nc, nn, "blue") == 2 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 4 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 5 ) ;
    assert( ssys::listed_count(nc, nn, "green") == 6 ) ;
    assert( ssys::listed_count(nc, nn, "purple") == -1 ) ;
    assert( ssys::listed_count(nc, nn, "blue") == 3 ) ;

    std::cout << ssys::desc_listed_count(nc, nn) << std::endl ;
}



void test_make_vec()
{
    const char* line = R"LITERAL(
    red
    green
    blue
    )LITERAL" ;

    std::vector<std::string>* v = ssys::make_vec<std::string>(line, '\n') ;
    int num_elem = v ? v->size() : 0 ;

    std::cout
        << "test_make_vec"
        << " num_elem " << num_elem
        << " line["
        << std::endl
        << line
        << std::endl
        ;

    for(int i=0 ; i < num_elem ; i++)
        std::cout << std::setw(3) << i << " : [" << (*v)[i] << "]" << std::endl ;
}

void test_getenv_vec_multiline()
{
    const char* fallback = R"LITERAL(
    red
    green
    blue
    )LITERAL" ;

    const char* ekey = "MULTILINE" ;

    //char delim = ',' ;
    char delim = '\n' ;

    std::vector<std::string>* v = ssys::getenv_vec<std::string>(ekey, fallback, delim) ;
    int num_elem = v ? v->size() : 0 ;

    const char* eval = getenv(ekey) ;

    std::cout
        << "test_getenv_vec_multiline"
        << std::endl
        << " ekey " << ekey
        << std::endl
        << " eval [" << ( eval ? eval : "-" ) << "]"
        << std::endl
        ;

    for(int i=0 ; i < num_elem ; i++)
        std::cout << std::setw(3) << i << " : [" << (*v)[i] << "]" << std::endl ;
}

void test_getenv_multiline()
{
    const char* ekey = "MULTILINE" ;
    const char* eval = getenv(ekey) ;

    std::cout << "test_getenv_multiline[" << std::endl ;
    std::cout << "[" << ( eval ? eval : "-" ) << "]" << std::endl ;
}

void test_fillvec()
{
    std::vector<int> vec ;
    ssys::fill_vec<int>(vec, "10,20,-30,40,50", ',' );
    assert( vec.size() == 5 && vec[0] == 10 && vec[4] == 50 );
}

void test_fill_vec()
{
    std::vector<int> vec ;
    ssys::fill_vec<int>(vec, "10,20,-30,40,50", ',' );
    assert( vec.size() == 5 && vec[0] == 10 && vec[4] == 50 );
}


template<typename T>
void test_fill_evec()
{
    std::vector<T> vec ;
    ssys::fill_evec<T>(vec, "EVEC", "10,20,-30,40,50", ',' );

    int num = vec.size() ;

    std::cout << "EVEC [" ;
    for(int i=0 ; i < num ; i++ ) std::cout << vec[i] << " " ;
    std::cout << " ] " << num << std::endl ;
}

void test_uname()
{
    std::vector<std::string> args = {"-n", "-a",""} ;
    int num = args.size() ;
    for(int i=0 ; i < num ; i++) std::cout
        << std::setw(4) << args[i]
        << " : "
        << ssys::uname(args[i].c_str())
        << std::endl
        ;
}


void test_fill_evec_string()
{
    std::vector<std::string> mat ;
    ssys::fill_evec(mat, "MAT", "Rock,Air,Water", ',' ) ;
    std::cout << ssys::desc_vec(&mat) << std::endl ;
}

void test_Dump()
{
    ssys::Dump("test_Dump");
    ssys::Dump("test_Dump");
    ssys::Dump("test_Dump");
    ssys::Dump("test_Dump");
}

void test_Desc()
{
    std::cout << ssys::Desc();
}
void test_PWD()
{
    std::cout << ssys::PWD();
}


void test_getenv_ParseInt()
{
    std::vector<std::string> _fallback = { "1", "10", "20", "M1" , "K1", "M2", "M3" } ;

    int num_fallback = _fallback.size();
    const int K = 1000 ;
    const int M = 1000000 ;
    const char* ekey = "HELLO" ;
    std::cout << "test_getenv_ParseInt ekey " << ekey << std::endl ;

    for(int i=0 ; i < num_fallback ; i++)
    {
        const char* fallback = _fallback[i].c_str();
        int num = ssys::getenv_ParseInt(ekey, fallback );

        std::cout
            << std::setw(20) << fallback
            << " num:   " << std::setw(10) << std::scientific << num
            << " num/M: " << std::setw(10) << num/M
            << " num/K: " << std::setw(10) << num/K
            << std::endl
            ;
    }
}



void test_getenv_with_prefix()
{
    typedef std::pair<std::string,std::string> KV ;

    std::vector<KV> kvs ;
    ssys::getenv_with_prefix(kvs, "OPTICKS_");

    int num_kvs = kvs.size() ;
    for(int i=0 ; i < num_kvs ; i++)
    {
        const KV& kv = kvs[i];
        std::cout
            << std::setw(3) << i << " "
            << "[" << kv.first << "]"
            << "[" << kv.second << "]"
            << std::endl
            ;
    }

}

void test_getenvintvec()
{
    const char* ekey = "CEGS" ;
    char delim = ':' ;
    const char* fallback = "16:0:9:1000" ;

    std::vector<int> cegs ;
    ssys::getenvintvec(ekey, cegs, delim, fallback );

    std::cout << "cegs[" ;
    for(int i=0 ; i < int(cegs.size()) ; i++) std::cout << cegs[i] << " " ;
    std::cout << "]" << std::endl ;

    assert( cegs.size() == 4 );
}

int test_getenvintspec()
{
    std::vector<std::string> specs = {{"0", "1", "2", "M1", "k1", "K1", "H1", "H10", "G1", "yellow", "magenta", "" }} ;
    const char* ekey = "test_getenvintspec_dummy" ;

    for(int i=0 ; i < int(specs.size()) ; i++)
    {
       const char* fallback = specs[i].c_str();
       int ival = ssys::getenvintspec(ekey, fallback ) ;
       std::cout
           << std::setw(4) << i
           << " "
           << " ssys::getenvintspec(\"" << ekey << "\",\"" << std::setw(10) << fallback << "\") "
           << " : "
           << ival
           << "\n"
           ;
    }
    return 0 ;
}











void test_getenvintpick()
{
    std::vector<std::string> colors = {{"red", "green", "blue", "cyan", "yellow", "magenta" }} ;
    int idx = ssys::getenvintpick("COLOR", colors, -1 );

    std::cout
       << "test_getenvintpick\n"
       << " idx " << idx << "\n"
       << " pick " << (( idx < int(colors.size()) && idx > -1 ) ? colors[idx] : "-" ) << "\n"
       ;
}




int main(int argc, char** argv)
{
    //const char* DEFT = "getenvintspec" ;
    const char* DEFT = "getenviron" ;
    const char* TEST = ssys::getenvvar("TEST", DEFT );
    bool ALL = strcmp("ALL", TEST) == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"popen_0")) rc += test_popen_0() ;
    if(ALL||0==strcmp(TEST,"popen_1")) rc += test_popen_1() ;
    if(ALL||0==strcmp(TEST,"username")) rc += test_username() ;
    if(ALL||0==strcmp(TEST,"is_remote_session")) rc += test_is_remote_session() ;
    if(ALL||0==strcmp(TEST,"getenvintspec"))  rc += test_getenvintspec() ;
    if(ALL||0==strcmp(TEST,"getenviron"))  rc += ssys_test::getenviron() ;
    if(ALL||0==strcmp(TEST,"getenvuint64spec"))  rc += ssys_test::getenvuint64spec() ;



    /*
    test_getenv_vec();
    test_getenv_();
    test_getenvvec_string_1();
    test_getenvvec_string_2();
    test__getenv();
    test_getenv_kv();
    test_replace_envvar_token();
    test_getenv_kv();
    test_getenvfloat();
    test_getenvdouble();
    test_which();
    test_getenvvar();
    test_is_listed();
    test_make_vec();
    test_getenv_multiline();
    test_getenv_vec_multiline();
    test_fill_vec();

    test_fill_evec<int>();
    test_fill_evec<float>();
    test_fill_evec<double>();
    test_fill_evec<std::string>();
    test_uname();
    test_listed_count();
    test_fill_evec_string();
    test_Dump();
    test_Desc();
    test_PWD();
    test_getenv_ParseInt();
    test_getenv_with_prefix();
    test_getenvintvec();
    test_getenvfloatvec_fallback();
    test_getenvintpick();
    test_getenvvar_path_replacement();
    */

    return rc ;
}

// ~/opticks/sysrap/tests/ssys_test.sh


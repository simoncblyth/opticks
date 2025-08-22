// ~/o/sysrap/tests/SGLM_test.sh

#include <functional>
#include "ssys.h"
#include "SGLM.h"

struct test_Assignment
{
    glm::tmat4x4<double> md  ;
    glm::tmat4x4<float>  mf  ;

    test_Assignment()
       :
       md(SGLM::DemoMatrix<double>(1.)),
       mf(SGLM::DemoMatrix<float>(2.f))
    {
        std::cout << "SGLM::DemoMatrix<double> md " << std::endl << SGLM::Present_<double>(md) << std::endl ;
        std::cout << "SGLM::DemoMatrix<float>  mf"  << std::endl << SGLM::Present_<float>(mf) << std::endl ;
    }
    static int Widening()
    {
        test_Assignment t ;
        t.md = t.mf ;
        std::cout << "SGLM::DemoMatrix<double> t.md (after t.md = t.mf    widening mf values into md)  " << std::endl << SGLM::Present_<double>(t.md) << std::endl ;
        return 0 ;
    }
    static int Narrowing()
    {
        test_Assignment t ;
        t.mf = t.md ;
        std::cout << "SGLM::DemoMatrix<float> t.mf (after t.mf = t.md    narrowing md values into mf)  " << std::endl << SGLM::Present_<float>(t.mf) << std::endl ;
        return 0 ;
    }
};


struct SGLM_test
{
    static int Dump();
    static int descBasis();
    static int descProjection();
    static int command();
    static int VIZMASK();
    static int UP();

    static int Main();
};


int SGLM_test::Dump()
{
    // setenv("WH", "1024,768", true );
    // setenv("CE", "0,0,0,100", true) ;
    // setenv("EYE", "-1,-1,0", true );
    // NB it is too late for setenv to influence SGLM as the static initialization would have happened already :
    // must use static methods to change the inputs that OR export envvars in the invoking script to configure defaults
    // SGLM::SetWH(1024,768);

    SGLM sglm ;
    sglm.dump();
    return 0 ;
}

int SGLM_test::descBasis()
{
    SGLM gm ;
    std::cout << gm.descBasis();

    float near_abs = 7.f ;
    float far_abs = 700.f ;
    gm.set_near_abs(near_abs);
    gm.set_far_abs(far_abs);

    std::cout << gm.descBasis();

    assert( gm.get_near_abs() == near_abs );
    assert( gm.get_far_abs() == far_abs );

    return 0 ;
}


int SGLM_test::descProjection()
{
    SGLM gm ;

    float near_abs = 7.f ;
    float far_abs = 700.f ;
    gm.set_near_abs(near_abs);
    gm.set_far_abs(far_abs);

    gm.update();

    std::cout << gm.descProjection();
    std::cout << gm.desc();
    std::cout << gm.desc_MV_P_MVP_ce_corners() ;

    //assert( gm.get_near_abs() == near_abs );
    //assert( gm.get_far_abs() == far_abs );

    return 0 ;
}






int SGLM_test::command()
{
    SGLM gm ;
    SCMD* cm = (SCMD*)&gm ;

    const char* cmd = "--ce 0,0,0,10 --eye 0,10,1 --look 0,1,1 --up 1,0,0 --zoom 5 --tmin 0.01 --tmax 1000" ;
    //const char* cmd = "--ce 0,0,0,10 --tmin 0.5 --tmax 5" ;

    int rc = cm->command(cmd);
    std::cout
        << " rc " << rc
        << std::endl
        ;

    std::cout << gm.desc() ;
    return 0 ;
}


int SGLM_test::VIZMASK()
{
    SGLM gm ;
    std::cout
        << "gm.vizmask "
        << gm.vizmask
        << "\n"
        //<< SBitSet
        ;

    return 0 ;
}

int SGLM_test::UP()
{
    std::cout
        << " SGLM::UP.x "
        <<   SGLM::UP.x
        << " SGLM::UP.y "
        <<   SGLM::UP.y
        << " SGLM::UP.z "
        <<   SGLM::UP.z
        << " SGLM::UP.w "
        <<   SGLM::UP.w
        << "\n"
        ;

    SGLM gm ;

    std::cout
        << " gm.UP.x "
        <<   gm.UP.x
        << " gm.UP.y "
        <<   gm.UP.y
        << " gm.UP.z "
        <<   gm.UP.z
        << " gm.UP.w "
        <<   gm.UP.w
        << "\n"
        ;

    return 0 ;
}




int SGLM_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST", "Dump") ;
    int rc = 0 ;
    if(strcmp(TEST, "Dump")==0 )           rc += Dump();
    if(strcmp(TEST, "descBasis")==0 )      rc += descBasis();
    if(strcmp(TEST, "descProjection")==0 ) rc += descProjection();
    if(strcmp(TEST, "command")==0 )        rc += command();
    if(strcmp(TEST, "VIZMASK")==0 )        rc += VIZMASK();
    if(strcmp(TEST, "Widening")==0 )       rc += test_Assignment::Widening();
    if(strcmp(TEST, "Narrowing")==0 )      rc += test_Assignment::Narrowing();
    if(strcmp(TEST, "UP")==0 )             rc += UP();

    return rc ;
}

int main(){ return SGLM_test::Main()  ; }



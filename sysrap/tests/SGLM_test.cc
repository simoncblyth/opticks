// ~/o/sysrap/tests/SGLM_test.sh 

#include <functional>
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
    static void Widening()
    {
        test_Assignment t ; 
        t.md = t.mf ;  
        std::cout << "SGLM::DemoMatrix<double> t.md (after t.md = t.mf    widening mf values into md)  " << std::endl << SGLM::Present_<double>(t.md) << std::endl ;  
    }
    static void Narrowing()
    {
        test_Assignment t ; 
        t.mf = t.md ;  
        std::cout << "SGLM::DemoMatrix<float> t.mf (after t.mf = t.md    narrowing md values into mf)  " << std::endl << SGLM::Present_<float>(t.mf) << std::endl ;  
    }
}; 


void test_SGLM()
{
    // setenv("WH", "1024,768", true ); 
    // setenv("CE", "0,0,0,100", true) ; 
    // setenv("EYE", "-1,-1,0", true ); 
    // NB it is too late for setenv to influence SGLM as the static initialization would have happened already : 
    // must use static methods to change the inputs that OR export envvars in the invoking script to configure defaults
    // SGLM::SetWH(1024,768); 

    SGLM sglm ; 
    sglm.dump();
}

void test_SGLM_basis()
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

}

void test_SGLM_command()
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


}


int main()
{
    /*
    test_Assignment::Widening(); 
    test_Assignment::Narrowing(); 
    test_SGLM_basis(); 
    test_SGLM(); 
    */
    test_SGLM_command(); 

    return 0 ; 
}

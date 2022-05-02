// name=SGLM_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name && /tmp/$name

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




int main()
{
    /*
    test_Assignment::Widening(); 
    test_Assignment::Narrowing(); 
    */

    test_SGLM(); 


    return 0 ; 
}

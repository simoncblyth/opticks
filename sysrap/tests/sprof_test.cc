#include "sprof.h"

void test_uninitialized()
{
    sprof u  ;  // uninitialized : starts with garbage 
    std::cout << " sprof u   : " << sprof::Desc(u) << std::endl ; 

    sprof* d0 = new sprof ; 
    std::cout << " new sprof : " << sprof::Desc(*d0) << std::endl ; 

    sprof* d1 = new sprof() ; 
    std::cout << " new sprof() :" << sprof::Desc(*d1) << std::endl ; 

}


void test_procedural()
{
    sprof p = {} ; // zeros 
    std::cout << sprof::Desc(p) << std::endl ; 

    for(int i=0 ; i < 100 ; i++)
    {
        sprof::Stamp(p); 

        std::string str = sprof::Desc(p) ; 
        bool llpt = sprof::LooksLikeProfileTriplet(str.c_str()) ; 
        std::cout << str << " llpt " << ( llpt ? "YES" : "NO " )  << std::endl ; 
    }
}


struct sprof_test
{
   sprof p0 ; 
   sprof p1 ;  

   void dump(const char* msg="") const ; 

   sprof_test(); 
   ~sprof_test(); 
}; 

void sprof_test::dump(const char* msg) const
{
    std::cout 
        << msg 
        << std::endl
        << "p0 " << sprof::Desc(p0) 
        << std::endl 
        << "p1 " << sprof::Desc(p1) 
        << std::endl  
        ;
}


sprof_test::sprof_test()
    :
    p0(),
    p1()
{
    sprof::Stamp(p0); 
    dump("ctor"); 
    sstamp::sleep(1); 
}

sprof_test::~sprof_test()
{
    sprof::Stamp(p1); 
    dump("dtor"); 
}


void test_ctor_dtor()
{
    sprof_test t ; 
}


int main(int argc, char** argv)
{
    //test_uninitialized(); 
    sprof_test t ; 
    return 0 ; 
}

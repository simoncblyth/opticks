#include "sprof.h"

void test_uninitialized()
{
    sprof u  ;  // uninitialized : starts with garbage 
    std::cout << " sprof u   : " << sprof::Desc_(u) << std::endl ; 

    sprof* d0 = new sprof ; 
    std::cout << " new sprof : " << sprof::Desc_(*d0) << std::endl ; 

    sprof* d1 = new sprof() ; 
    std::cout << " new sprof() :" << sprof::Desc_(*d1) << std::endl ; 

}


void test_procedural()
{
    sprof p = {} ; // zeros 
    std::cout << sprof::Desc_(p) << std::endl ; 

    for(int i=0 ; i < 100 ; i++)
    {
        sprof::Stamp(p); 

        std::string str = sprof::Desc_(p) ; 
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
        << "p0 " << sprof::Desc_(p0) 
        << std::endl 
        << "p1 " << sprof::Desc_(p1) 
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
void test_Serialize_Import()
{
    sprof p0 ; 
    sprof::Stamp(p0); 



    std::string data = sprof::Serialize(p0); 
    std::cout << data << std::endl ; 

    sprof p1 ; 
    sprof::Import( p1, data.c_str() ) ; 

    std::cout << "p0:" << sprof::Desc_(p0)  << std::endl ; 
    std::cout << "p1:" << sprof::Desc_(p1)  << std::endl ; 

    assert( sprof::Equal(p0, p1) ); 
}


int main(int argc, char** argv)
{
    /*
    test_uninitialized(); 
    test_ctor_dtor();
    */

    test_Serialize_Import(); 
    
    return 0 ; 
}

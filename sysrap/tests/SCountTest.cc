#include "OPTICKS_LOG.hh"
#include "SCount.hh"

void test_basic()
{
    SCount cn ;

    cn.add(1) ; 
    cn.add(1) ; 
    cn.add(1) ; 

    cn.add(42) ; 
    cn.add(42) ; 

    cn.add(1042) ; 
    cn.add(1042) ; 
    cn.add(1042) ; 


    LOG(info) << cn.desc() ; 
}


void test_is_all()
{
    SCount cn ;
    cn.add(1) ; 
    cn.add(10) ; 
    cn.add(100) ; 
   
    assert( cn.is_all(1) == true ); 

    cn.add(100) ; 
    assert( cn.is_all(1) == false ); 

    cn.add(1); 
    cn.add(10); 

    assert( cn.is_all(2) == true ); 

    LOG(info) << cn.desc() ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    test_basic(); 
    test_is_all(); 

    return 0 ; 
}

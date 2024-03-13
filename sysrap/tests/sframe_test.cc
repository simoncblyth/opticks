/**
sframe_test.cc
===============

::

   ~/o/sysrap/tests/sframe_test.sh

**/


#include "sframe.h"

void test_really_uninitialized_dtor()
{
    sframe fr ; 
}


void test_dtor_after_copy_for_double_free()
{
    sframe a ; 
    a.setTranslate(100.f, 200.f, 300.f) ; 
    a.prepare(); 

    sframe b = a ; 

    /**
    BEFORE ADDING COPY CTOR THAT RUNS prepare 

    assert( b.tr_m2w == a.tr_m2w ); 
    assert( b.tr_w2m == a.tr_w2m ); 
    //here is the cause of double free : b thinks it owns the pointers of a 
    **/

    // AFTER ADDING COPY CTOR THAT RUNS prepar

    assert( b.tr_m2w != a.tr_m2w ); 
    assert( b.tr_w2m != a.tr_w2m ); 

}



void test_uninitialized()
{
    sframe fr = {} ; 
    std::cout << "fr" << std::endl << fr << std::endl ; 

    std::cout << "fr.m2w.q3.i.w " << fr.m2w.q3.i.w  << std::endl; 
    std::cout << "fr.m2w.q3.f.w " << fr.m2w.q3.f.w  << std::endl; 
}


void test_save_load()
{
    sframe a ; 
    a.frs = "a test of persisting via metadata" ;
    a.ce.x = 1.f ; 
    a.ce.y = 2.f ; 
    a.ce.z = 3.f ; 
    a.ce.w = 4.f ; 
    std::cout << "a" << std::endl << a << std::endl ; 
    a.save("$FOLD"); 

    sframe b = sframe::Load("$FOLD");  
    std::cout << "b" << std::endl << b << std::endl ; 
    assert( strcmp(a.frs, b.frs) == 0 ); 
}

void test_load()
{
    sframe b = sframe::Load("$FOLD");  
    std::cout << "b" << std::endl << b << std::endl ; 
}

void test_setTranslate()
{
    sframe fr ; 
    fr.setTranslate(100.f, 200.f, 300.f) ; 
    fr.prepare(); 

    std::cout << fr << std::endl ; 
    fr.save("$FOLD"); 
}


int main(int argc, char** argv)
{
    //test_really_uninitialized_dtor(); 
    test_dtor_after_copy_for_double_free(); 

    /*
    test_uninitialized(); 
    test_save_load() ; 
    test_load(); 
    test_setTranslate(); 
    */

    return 0 ; 
}

// ./s_pool_test.sh 

/**
s_pool_test.cc
=================



**/

#include "Obj.h"
Obj::POOL Obj::pool = {} ; 


void test_stack_Obj()
{
    Obj o(100); 
}
void test_heap_Obj()
{
    Obj* o = new Obj(100); 
    delete o ; 
}

void test_vector_0()
{
    printf("//test_vector_0\n"); 
    Obj a(100); 
    Obj b(200); 
    Obj c(300); 

    std::vector<Obj> oo = {a, b, c } ; 

    // Obj dtor runs twice here because its being copied into the vector ? 

/*
//test_vector_0
s_pool::add pid 0
s_pool::add pid 1
s_pool::add pid 2
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove failed to find the object : already removed, double dtors ?  
s_pool::remove pid 2
s_pool::remove pid 1
s_pool::remove pid 0
*/

    // HMM: guess the pool needs to be aware of copy ctor to avoid this 
}

void test_vector_1()
{
    Obj a(100); 
    Obj b(200); 
    Obj c(300); 

    std::vector<Obj> oo ; 
    oo.push_back(a); 
    oo.push_back(b); 
    oo.push_back(c); 
    // same as _0 
}



void test_roundtrip()
{
    std::vector<_Obj> buf ; 
    {
        Obj* a = new Obj(100) ; 
        Obj* b = new Obj(200) ; 
        Obj c(300, a, b );     

        // because Obj deep deletes would get double delete 
        // with a and b on stack 
        Obj::pool.serialize(buf) ; 
    }

    std::cout << " buf.size " << buf.size() << std::endl ; 

    Obj::pool.import(buf); 
}



int main()
{
    /*
    test_vector_0(); 
    test_vector_1(); 
    test_heap_Obj(); 
    test_stack_Obj(); 
    test_roundtrip();
    */
    test_roundtrip();

    return 0 ; 
}

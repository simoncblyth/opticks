// ./s_pool_test.sh 

/**
s_pool_test.cc
=================



**/

#include "Obj.h"


struct s_pool_test
{
    static int stack_Obj(); 
    static int heap_Obj(); 
    static int vector_0_FAILS(); 
    static int vector_1_FAILS(); 
    static int vector_2(); 
    static int create_delete_0(); 
    static int roundtrip(); 
    static int main(); 
};


int s_pool_test::stack_Obj()
{
    Obj o(100); 
    return 0 ; 
}
int s_pool_test::heap_Obj()
{
    Obj* o = new Obj(100); 
    delete o ; 
    return 0 ; 
}

/**
s_pool_test::vector_0_FAILS
------------------------------

Trying to use stack objects causes fail
because of double dtors 

Obj dtor runs doubled up because its being copied into the vector, 
or due to temporaries perhaps.  

**/


int s_pool_test::vector_0_FAILS()
{
    printf("//test_vector_0_FAILS\n"); 
    Obj a(100); 
    Obj b(200); 
    Obj c(300); 

    std::vector<Obj> oo = {a, b, c } ; 

    return 0 ; 
}

int s_pool_test::vector_1_FAILS()
{
    Obj a(100); 
    Obj b(200); 
    Obj c(300); 

    std::vector<Obj> oo ; 
    oo.push_back(a); 
    oo.push_back(b); 
    oo.push_back(c); 
    // same as _0 
    return 0 ; 
}


int s_pool_test::vector_2()
{
    Obj* a = new Obj(100); 
    Obj* b = new Obj(200); 
    Obj* c = new Obj(300); 

    std::vector<Obj*> oo ; 
    oo.push_back(a); 
    oo.push_back(b); 
    oo.push_back(c); 

    return 0 ; 
}

/**
s_pool_test::create_delete_0
------------------------------

1. create 10 then delete the first 5 
2. check that Obj::Index for the remaining pointers
   gives the expected 0,1,2,3,4


::

   0.   X
   1.   X
   2.   X
   3.   X  
   4.   X
   5.       0
   6.       1
   7.       2
   8.       3
   9.       4


**/



int s_pool_test::create_delete_0()
{
    static int N1 = 10 ; 
    static int D1 = 5  ; 
    Obj* oo[N1] ;

    for(int i=0 ; i < N1 ; i++) 
    {
        Obj* o = new Obj(100.) ; 
        oo[i] = o ;
    }
    for(int i=0 ; i < N1 ; i++) 
    {
        Obj* o0 = oo[i] ; 

        int idx = Obj::Index(o0) ; 
        assert( idx == i ) ;  
  
        Obj* o1 = Obj::GetByIdx(i); 
        assert( o1 == o0 ); 
    }


    for(int i=0 ; i < D1 ; i++) 
    {
        Obj* o = oo[i] ;   
        delete o ; 
        oo[i] = nullptr ; 
    }  



    for(int i=0 ; i < N1 ; i++) 
    {
        Obj* o0 = oo[i] ; 
        int idx = o0 == nullptr ? -1 : Obj::Index(o0) ; 
        int x_idx = i < D1 ? -1 : i - D1 ;  
        assert( idx == x_idx ) ;  
    }



    for(int i=0 ; i < N1 ; i++) 
    {
        int pid = i ; 
        Obj* o = Obj::Lookup(pid) ; 
        std::cout << "Obj::Lookup(" << std::setw(2) << pid << ") : " << Obj::Desc(o) << "\n" ; 
        Obj* o_x = pid < D1 ? oo[D1+pid] : nullptr ;  
        //assert( o == o_x ) ;  
    }


    for(int i=0 ; i < N1 ; i++) 
    {
        int idx = i ; 
        Obj* o = Obj::GetByIdx(idx) ; 
        std::cout << "Obj::GetByIdx(" << std::setw(2) << i << ") : " << Obj::Desc(o) << "\n" ; 
        Obj* o_x = idx < D1 ? oo[D1+idx] : nullptr ;  
        assert( o == o_x ) ;  
    }




    return 0 ; 
}







int s_pool_test::roundtrip()
{
    std::vector<_Obj> buf ; 
    {
        Obj* a = new Obj(100) ; 
        Obj* b = new Obj(200) ; 
        Obj c(300, a, b );     

        // because Obj deep deletes would get double delete 
        // with a and b on stack 
        Obj::pool->serialize_(buf) ; 
    }

    std::cout << " buf.size " << buf.size() << std::endl ; 

    Obj::pool->import_(buf); 

    return 0 ; 
}


int s_pool_test::main()
{
    const char* TEST = ssys::getenvvar("TEST","vector_0") ; 

    int rc = 0 ; 
    if(      strcmp(TEST, "vector_0_FAILS") == 0 ) rc = vector_0_FAILS() ;
    else if( strcmp(TEST, "vector_1_FAILS") == 0 ) rc = vector_1_FAILS() ;
    else if( strcmp(TEST, "vector_2") == 0 ) rc = vector_2() ;
    else if( strcmp(TEST, "vector_2") == 0 ) rc = vector_2() ;
    else if( strcmp(TEST, "create_delete_0") == 0 ) rc = create_delete_0() ;
    else if( strcmp(TEST, "heap_Obj") == 0 ) rc = heap_Obj();
    else if( strcmp(TEST, "stack_Obj") == 0 ) rc = stack_Obj();
    else if( strcmp(TEST, "roundtrip") == 0 ) rc = roundtrip();
 
    return rc ; 
}


int main(int argc, char** argv)
{
    Obj::pool = new Obj::POOL("Obj") ;  

    return s_pool_test::main() ; 
}




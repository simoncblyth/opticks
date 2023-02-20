#include "stv.h"
stv::POOL stv::pool = {} ; 

const char* FOLD = getenv("FOLD"); 

void test_serialize()
{
    stv* a = new stv ; a->t[0][0] = 100. ; 
    stv* b = new stv ; b->v[0][0] = 200. ; 
    if(stv::pool.level > 0) std::cout << stv::pool.desc() ;  

    NP* arr = stv::pool.serialize<double>() ;  
    arr->save(FOLD, stv::NAME ); 

    delete a ; 
    delete b ; 
}

void test_import()
{
    NP* arr = NP::Load(FOLD, stv::NAME) ; 
    stv::pool.import<double>(arr) ; 
    if(stv::pool.level > 0) std::cout << stv::pool.desc() ;  
}

int main()
{
    //stv::pool.level = 2 ; 
    test_serialize(); 
    test_import(); 
    return 0 ; 
}

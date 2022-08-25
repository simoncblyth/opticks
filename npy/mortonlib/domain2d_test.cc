// ./domain2d_test.sh 

#include <cstdlib>
#include "domain2d.h"

const char* FOLD = getenv("FOLD"); 

void morton_circle_demo()
{
    domain2d dom( -100, 100, -100, 100 ); 
    std::vector<uint64_t> kk ;
    dom.get_circle(kk, 50.f ); 

    uint64_t mask = ~0xfffffff ;  // 7 nibbles flipped  
    NP* a = dom.make_array( kk, mask ); 

    a->save(FOLD, "morton_circle_demo.npy"); 
    std::cout << " save " << FOLD << " " << a->sstr() << std::endl ; 
}


int main(int argc, char** argv)
{
    morton_circle_demo();  

    return 0 ; 
}


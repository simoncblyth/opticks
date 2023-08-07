// ./csg_intersect_leaf_templated_test.sh

#include <cstdio>
#include "scuda.h"
#include "squad.h"
#include "scuda_templated.h"

#include "csg_intersect_leaf_templated.h"


template<typename F>
void test_distance_leaf_sphere()
{
    F3 pos = {0.f, 0.f, 0.f} ;  

    Q4 q ;  
    q.f = {0.f, 0.f, 0.f, 100.f } ; 

    F t = distance_leaf_sphere<F>( pos, q ); 

    printf("//test_distance_leaf_sphere t %7.4f \n", t ); 
}


int main()
{
    test_distance_leaf_sphere<float>(); 
    test_distance_leaf_sphere<double>(); 

    return 0 ; 

}

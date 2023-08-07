/**
scuda_templated_test.cc : explore how to implement templated functions using float3/double3 etc..
=====================================================================================================


**/

#include <iostream>
#include "scuda.h"
#include "scuda_double.h"
#include "squad.h"
#include "scuda_templated.h"

template <typename F>
F dummy_intersect( const F3& pos, const F3& dir )
{
    std::cout << " pos " << pos << std::endl ;      
    std::cout << " dir " << dir << std::endl ;      
    return 0.f  ; 
}

template <typename F>
void check_dummy_intersect()
{
     std::cout << "check<" << ( sizeof(F) == 4 ? "float" : "double" ) << ">" << std::endl  ; 

     F3 pos = {0.f, 0.f, 0.f} ; 
     F3 dir = {0.f, 0.f, 1.f} ; 
     F dist = dummy_intersect<F>( pos, dir ) ; 
     std::cout << " dist " << dist << std::endl ; 

     Q4 q ; 
     q.u = {0,1,2,3} ; 

     std::cout << q << std::endl ; 
}



int main()
{
     check<float>(); 
#ifdef WITH_SCUDA_DOUBLE
     check<double>(); 
#endif

     return 0 ; 
}

// ./snd_test.sh

#include <iostream>
#include <vector>
#include "snd.h"

int main(int argc, char** argv)
{
    snd a = {} ; 
    a.setParam( 1., 2., 3., 4., 5., 6. ); 

    snd b = snd::Sphere(100.) ; 
    snd c = snd::ZSphere(100., -10.,  10. ) ; 

    std::vector<snd> nn = {a, b, c } ;

    for(unsigned i=0 ; i < nn.size() ; i++) std::cout << nn[i] ;  

    return 0 ; 
}


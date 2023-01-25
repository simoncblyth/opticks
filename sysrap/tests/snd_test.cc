// name=snd_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <vector>

#include "snd.h"

int main(int argc, char** argv)
{
    snd<double> a = {} ; 
    a.setParam( 1., 2., 3., 4., 5., 6. ); 

    snd<double> b = snd<double>::Sphere(100.) ; 
    snd<double> c = snd<double>::ZSphere(100., -10.,  10. ) ; 

    std::vector<snd<double>> nn = {a, b, c } ;

    for(unsigned i=0 ; i < nn.size() ; i++) std::cout << nn[i] ;  

    return 0 ; 
}


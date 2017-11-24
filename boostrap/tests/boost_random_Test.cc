#include <iostream>

//#include <boost/random/mersenne_twister.hpp>
#include <boost/random/inversive_congruential.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>


#include "PLOG.hh"

using std::cout ; 
using std::endl  ; 


void test_0()
{

    boost::mt19937 gener(1);
    boost::normal_distribution<> normal(0,1);
    boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > rng(gener, normal);

    cout << rng() << endl;
    cout << rng() << endl;
    cout << rng() << endl;

    gener.seed(2);
    cout << rng() << endl;
    cout << rng() << endl;
    gener.seed(1);
    cout << rng() << endl;
    gener.seed(2);
    cout << rng() << endl;
    gener.seed(3);
    cout << rng() << endl;

}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_0();

    return 0 ; 
}



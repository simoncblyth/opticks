/**

~/o/sysrap/tests/s_unique_test.sh

**/


#include "s_unique.h"
#include <iostream>

int main()
{
    std::vector<std::string> unam = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten"
    };


    std::vector<int> val = { 0,1,2, 3,1,1,1, 4,2,6,2,10, 3,3,3,6,3,3, 4,4,5,4,4,4, 7, 8, 9 } ;
    std::vector<int> uval ; 

    std::vector<std::size_t> count ;
    std::vector<std::size_t> order ;
    std::vector<std::size_t> index ;
    std::vector<std::size_t> inverse ;
    std::vector<int> original ;


    s_unique(uval, val.begin(),   val.end(), &count, &order, &index, &inverse, &original );  
    std::cout << s_unique_desc( uval, &unam, &count, &order, &index, &inverse, &original ) ;

    return 0 ; 

}

#include "strided_range.h"

#include <thrust/fill.h>
#include <thrust/copy.h>
#include <ostream>

int main(void)
{
    thrust::device_vector<int> data(8);
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;
    data[5] = 60;
    data[6] = 70;
    data[7] = 80;

    // print the initial data
    std::cout << "data: ";
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    typedef thrust::device_vector<int>::iterator Iterator;
    
    // create strided_range with indices [0,2,4,6]
    strided_range<Iterator> evens(data.begin(), data.end(), 2);
    std::cout << "sum of even indices: " << thrust::reduce(evens.begin(), evens.end()) << std::endl;
    
    // create strided_range with indices [1,3,5,7]
    strided_range<Iterator> odds(data.begin() + 1, data.end(), 2);
    std::cout << "sum of odd indices:  " << thrust::reduce(odds.begin(), odds.end()) << std::endl;

    // set odd elements to 0 with fill()
    std::cout << "setting odd indices to zero: ";
    thrust::fill(odds.begin(), odds.end(), 0);
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    return 0;
}
/*
simon:thrustrap blyth$ /usr/local/env/numerics/thrustrap/bin/strided_rangeTest
data: 10 20 30 40 50 60 70 80 
sum of even indices: 160
sum of odd indices:  200
setting odd indices to zero: 10 0 30 0 50 0 70 0 
*/



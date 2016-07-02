// /usr/local/env/numerics/thrust/examples/repeated_range.cu
#include "repeated_range.h"
//#include <thrust/fill.h>

// for printing
#include <thrust/copy.h>
#include <ostream>


int main(void)
{
    thrust::device_vector<int> data(4);
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;

    // print the initial data
    std::cout << "range        ";
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    typedef thrust::device_vector<int>::iterator Iterator;
  
    // create repeated_range with elements repeated twice
    repeated_range<Iterator> twice(data.begin(), data.end(), 2);
    std::cout << "repeated x2: ";
    thrust::copy(twice.begin(), twice.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;
    
    // create repeated_range with elements repeated x3
    repeated_range<Iterator> thrice(data.begin(), data.end(), 3);
    std::cout << "repeated x3: ";
    thrust::copy(thrice.begin(), thrice.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;

    return 0;
}

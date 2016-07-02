#include "strided_repeated_range.h"

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
    
    strided_repeated_range<Iterator> sr(data.begin(), data.end(), 2, 3);
    thrust::copy(sr.begin(), sr.end(), std::ostream_iterator<int>(std::cout, " "));  std::cout << std::endl;


    return 0;

}


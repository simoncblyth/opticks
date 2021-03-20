/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "THRAP_HEAD.hh"
#include "strided_range.h"

#include <thrust/fill.h>
#include <thrust/copy.h>
#include "THRAP_TAIL.hh"

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



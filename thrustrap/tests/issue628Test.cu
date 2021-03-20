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


//  https://github.com/thrust/thrust/issues/628

#include "THRAP_HEAD.hh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
#include "THRAP_TAIL.hh"
#include <iostream>

int main(int argc, char ** argv)
{
    thrust::device_vector<int> indices(10);
    thrust::sequence(indices.begin(), indices.end());

    thrust::device_vector<int> temp(10, -1);

    thrust::counting_iterator<int> iter(0);
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(iter, iter)),
                          thrust::make_zip_iterator(thrust::make_tuple(iter, iter)) + temp.size(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          temp.begin(),
                          thrust::equal_to<thrust::tuple<int,int> >(),
                          thrust::plus<int>());

    std::copy(temp.begin(), temp.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    thrust::fill(temp.begin(), temp.end(), -1);

    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(indices.begin(), indices.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(indices.end(), indices.end())),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          temp.begin(),
                          thrust::equal_to<thrust::tuple<int,int> >(),
                          thrust::plus<int>());

    std::copy(temp.begin(), temp.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}

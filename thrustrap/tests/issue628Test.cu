
//  https://github.com/thrust/thrust/issues/628

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/discard_iterator.h>
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

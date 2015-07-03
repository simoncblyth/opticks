#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "strided_repeated_range.h"


template<typename S>
void repeat_to_imp( 
                    unsigned int stride, 
                    unsigned int repeat, 
                    const thrust::device_vector<S>&  source,
                          thrust::device_vector<S>&  target)
{

    typedef typename thrust::device_vector<S>::const_iterator Iterator;

    strided_repeated_range<Iterator> sr(source.begin(), source.end(), stride, repeat);

    thrust::copy( sr.begin(), sr.end(),  target.begin() );    
}



template void repeat_to_imp<unsigned char>( 
                                                   unsigned int stride, 
                                                   unsigned int repeat, 
                    const thrust::device_vector<unsigned char>&  source,
                          thrust::device_vector<unsigned char>&  target
                                          );

template void repeat_to_imp<unsigned long long>( 
                                                   unsigned int stride, 
                                                   unsigned int repeat, 
                    const thrust::device_vector<unsigned long long>&  source,
                          thrust::device_vector<unsigned long long>&  target
                                          );


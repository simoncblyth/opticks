#pragma once
//
//   See *strided_repeated_range.py* to understand this, 
//   repeating is done "inside" the striding 
//   for example allowing a photon array to be duped up
//   to become a record array  (there being maxrec eg 10 records for every photon)
//   but retaining item structure, typically quads.
//
//   For example in the below strided_repeat the item  [0,1] 
//   is duplicated three times before the stride gets to the 2nd item [2,3]
//
//
//                                                  0  1  2  3  4  5  6  7  8  9  10 11
//   strided_repeated_range([0, 1, 2, 3], 2, 3) -> [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3] 
//
//
//       4*3 = 12   (striding doesnt effect size, just ordering)
//
//         stride: 2     ("itemsize")   
//        repeats: 3    
//
//    for an input index from i from 0 to N-1     N = (end - begin)*repeats
//    supply the index that does the required "permutation"
//
//         repeat  :     i/repeats        integer arithmetic
//         stride  :     i * stride
//   stride_repeat :     stride*(i/stride_repeats) + (i % stride)
//
//
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//                                   

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include "assert.h"


template <typename Iterator>
class strided_repeated_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_repeat_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride  ;
        difference_type repeats ;
        difference_type stride_repeats ;

        stride_repeat_functor(difference_type stride, difference_type repeats)
            : 
            stride(stride), 
            repeats(repeats),
            stride_repeats(stride*repeats) 
            {
            }

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride*(i/stride_repeats) + (i % stride);
        }

    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_repeat_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    typedef PermutationIterator iterator;

    strided_repeated_range(Iterator first, Iterator last, difference_type stride, difference_type repeats)
        : 
        first(first), 
        last(last), 
        stride(stride),
        repeats(repeats) 
    {
        //assert( (last - first) % stride == 0 && "restricting stride to fit exactly into the range") ;  
    }
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_repeat_functor(stride, repeats)));
    }

    iterator end(void) const
    {
        return begin() + repeats * (last - first);  // stride drops out 
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride ;
    difference_type repeats;
};



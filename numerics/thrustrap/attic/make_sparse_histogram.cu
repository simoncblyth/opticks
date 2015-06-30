#include "make_sparse_histogram.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
//#include <thrust/binary_search.h>
//#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "strided_range.h"

#include <iostream>
#include <iomanip>
#include <iterator>


template <typename Vector>
void print_vector_strided(const std::string& name, const Vector& v, bool hex=false, unsigned int stride=1)
{
  typedef typename Vector::value_type T;

  std::cout << "print_vector_strided (" << stride << ") " << std::setw(20) << name << std::endl  ;
  if(hex) std::cout << std::hex ; 
   
  //typedef thrust::device_vector<T>::iterator Iterator;
  typedef typename Vector::const_iterator Iterator;
  strided_range<Iterator> sv(v.begin(), v.end(), stride);

  thrust::copy(sv.begin(), sv.end(), std::ostream_iterator<T>(std::cout, " "));

  std::cout << std::endl;
  if(hex) std::cout << std::dec ; 
}



template <typename Vector>
void print_vector(const std::string& name, const Vector& v, bool hex=false, bool total=false)
{
  typedef typename Vector::value_type T;

  std::cout << "print_vector " << " : " << std::setw(40) << name << " " ;
  if(hex) std::cout << std::hex ; 
   
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));

  if(total)
  {
      T tot = thrust::reduce(v.begin(), v.end());
      std::cout << " total: " << tot ;
  }
   
  std::cout << std::endl;


  if(hex) std::cout << std::dec ; 
}



// sparse histogram using reduce_by_key
template <typename Vector1,
          typename Vector2,
          typename Vector3,
          typename Vector4>
void sparse_histogram(const Vector1& input,
                            Vector2& histogram_values,
                            Vector3& histogram_counts,
                            Vector4& histogram_index)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector3::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);

  unsigned int stride = 1000 ; 

  print_vector_strided("initial data", data, true, stride );

  thrust::sort(data.begin(), data.end());
  
  print_vector_strided("sorted data", data, true, stride );


  // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
  IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             IndexType(1),
                                             thrust::plus<IndexType>(),
                                             thrust::not_equal_to<ValueType>());

  // resize histogram storage
  histogram_values.resize(num_bins);
  histogram_counts.resize(num_bins);
  histogram_index.resize(num_bins);
  
  // compact find the end of each bin of values
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<IndexType>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  
  // print the sparse histogram

  print_vector("histogram values", histogram_values, true);
  print_vector("histogram counts", histogram_counts, false, true );

  
  thrust::sort_by_key( histogram_counts.begin(), histogram_counts.end(), 
                       histogram_values.begin());
                

  //thrust::sequence( histogram_index.begin(), histogram_index.end() );
     
  print_vector("histogram values (sorted by counts)", histogram_values, true);
  print_vector("histogram counts (sorted by counts)", histogram_counts, false, true);
  //print_vector("histogram index                    ", histogram_index , false);


  //  http://docs.thrust.googlecode.com/hg/group__replacing.html
  //
  // hmm need to replace the data values with indices 

}


#ifdef SPARTAN
void make_sparse_histogram(History_t* data, unsigned int numElements, Flags* flags )
#else
void make_sparse_histogram(History_t* data, unsigned int numElements, Types* flags )
#endif
{

    thrust::host_vector<History_t> input(data, data+numElements);
 
    std::cout << "Sparse Histogram" << std::endl;

    thrust::device_vector<History_t> histogram_values;
    thrust::device_vector<int>       histogram_counts;
    thrust::device_vector<int>       histogram_index;

    sparse_histogram(input, 
                     histogram_values, 
                     histogram_counts,
                     histogram_index
                    );


    thrust::host_vector<History_t> values = histogram_values ; 
    thrust::host_vector<int>       counts = histogram_counts ; 

    for(unsigned int i=0 ; i < values.size() ; i++)
    {
        History_t seq = values[i];
        std::string sseq = flags ? flags->getSequenceString(seq) : "" ; 
        std::cout << std::setw(5) << i 
                  << std::setw(20) << std::hex << seq
                  << std::setw(20) << std::dec << counts[i]
                  << "  " << sseq
                  << std::endl ; 
    }







}



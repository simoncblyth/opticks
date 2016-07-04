# === func-gen- : numerics/thrust/thrustexamples/thrustexamples fgp numerics/thrust/thrustexamples/thrustexamples.bash fgn thrustexamples fgh numerics/thrust/thrustexamples
thrustexamples-src(){      echo numerics/thrust/thrustexamples/thrustexamples.bash ; }
thrustexamples-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(thrustexamples-src)} ; }
thrustexamples-vi(){       vi $(thrustexamples-source) ; }
thrustexamples-usage(){ cat << EOU

* https://github.com/thrust/thrust/tree/master/examples


repeated_range
      duplicate elements n times 






::

    simon:examples blyth$ thrustexamples-nvcc
    nvcc /usr/local/env/numerics/thrust/examples/repeated_range.cu -o /usr/local/env/numerics/thrustexamples/repeated_range
    range        10 20 30 40 
    repeated x2: 10 10 20 20 30 30 40 40 
    repeated x3: 10 10 10 20 20 20 30 30 30 40 40 40 
    simon:examples blyth$ 


    simon:examples blyth$ thrustexamples-
    simon:examples blyth$ thrustexamples-nvcc histogram
    nvcc /usr/local/env/numerics/thrust/examples/histogram.cu -o /usr/local/env/numerics/thrustexamples/histogram
    Dense Histogram
              initial data  3 4 3 5 8 5 6 6 4 4 5 3 2 5 6 3 1 3 2 3 6 5 3 3 3 2 4 2 3 3 2 5 5 5 8 2 5 6 6 3 
               sorted data  1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8 
      cumulative histogram  0 1 7 19 23 32 38 38 40 
                 histogram  0 1 6 12 4 9 6 0 2 
    Sparse Histogram
              initial data  3 4 3 5 8 5 6 6 4 4 5 3 2 5 6 3 1 3 2 3 6 5 3 3 3 2 4 2 3 3 2 5 5 5 8 2 5 6 6 3 
               sorted data  1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8 
          histogram values  1 2 3 4 5 6 8 
          histogram counts  1 6 12 4 9 6 2 
    simon:examples blyth$ 



::

    107 void sparse_histogram(const Vector1& input,
    108                             Vector2& histogram_values,
    109                             Vector3& histogram_counts)
    110 {
    111   typedef typename Vector1::value_type ValueType; // input value type
    112   typedef typename Vector3::value_type IndexType; // histogram index type
    113 
    114   // copy input data (could be skipped if input is allowed to be modified)
    115   thrust::device_vector<ValueType> data(input);
    116 
    117   // print the initial data
    118   print_vector("initial data", data);
    119 
    120   // sort data to bring equal elements together
    121   thrust::sort(data.begin(), data.end());
    122 
    123   // print the sorted data
    124   print_vector("sorted data", data);
    125 
    126   // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
    127   IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
    128                                              data.begin() + 1,
    129                                              IndexType(1),
    130                                              thrust::plus<IndexType>(),
    131                                              thrust::not_equal_to<ValueType>());
    /// 
    ///   http://docs.thrust.googlecode.com/hg/group__transformed__reductions.html
    ///
    ///   generalized inner product with user provided "+" and "*"
    ///   shunting both ways allows to look for "edges" via "*" replacement of not_equal_to
    ///    
    ///     lop-off-end         1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 X  
    ///     lop-off-start     X 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8 
    ///                         .           .                       .       .                 .           .
    ///                         1           2                       3       4                 5           6    (+1 the init) -> 7 bins
    ///
    133   // resize histogram storage
    134   histogram_values.resize(num_bins);
    135   histogram_counts.resize(num_bins);
    136 
    137   // compact find the end of each bin of values
    138   thrust::reduce_by_key(data.begin(), 
                                data.end(),
    139                         thrust::constant_iterator<IndexType>(1),
    140                         histogram_values.begin(),
    141                         histogram_counts.begin());
    142 
    ///
    ///    http://docs.thrust.googlecode.com/hg/group__reductions.html#ga1fd25c0e5e4cc0a6ab0dcb1f7f13a2ad
    ///
    ///    thrust::pair<OutputIterator1,OutputIterator2> thrust::reduce_by_key (   
    ///          InputIterator1  keys_first,
    ///          InputIterator1  keys_last,
    ///          InputIterator2  values_first,   <--- using always 1 
    ///          OutputIterator1 keys_output,
    ///          OutputIterator2 values_output 
    ///    )       
    ///
    ///     reduce_by_key is a generalization of reduce to key-value pairs. 
    ///
    ///     For each group of consecutive keys in the range [keys_first, keys_last) that are equal, 
    ///         * reduce_by_key copies the first element of the group to the keys_output. 
    ///         * The corresponding values in the range are reduced using the plus 
    ///           and the result copied to values_output.
    ///
    ///          data       1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8
    ///          keys_out   1 2           3                       4       5                 6           8 
    ///          values_out 1 6          12                       4       9                 6           2
    ///
    ///
    ///
    ///
    ///



Adding sorting by count::

     72   thrust::sort_by_key( histogram_counts.begin(), histogram_counts.end(),
     73                        histogram_values.begin());
     74 
     75   print_vector("histogram values (sorted by counts)", histogram_values);
     76   print_vector("histogram counts (sorted bt counts)", histogram_counts);



::

    simon:thrust blyth$ nvcc histogram_sorted.cu -o /tmp/histogram_sorted
    simon:thrust blyth$ /tmp/histogram_sorted
    Sparse Histogram
              initial data  3 4 3 5 8 5 6 6 4 4 5 3 2 5 6 3 1 3 2 3 6 5 3 3 3 2 4 2 3 3 2 5 5 5 8 2 5 6 6 3 
               sorted data  1 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 4 4 4 4 5 5 5 5 5 5 5 5 5 6 6 6 6 6 6 8 8 
          histogram values  1 2 3 4 5 6 8 
          histogram counts  1 6 12 4 9 6 2 
      histogram values (sorted by counts)  1 8 4 2 6 5 3 
      histogram counts (sorted bt counts)  1 2 4 6 6 9 12 
    simon:thrust blyth$                                
                                           7 6 5 4 3 2 1   <-- count ordered index 


Hmm how to conjure values by their bin index in the histogram eg::

              initial data  3 4 3 5 8 5 6 6 4 4 5 3 2 5 6 3 1 3 2 3 6 5 3 3 3 2 4 2 3 3 2 5 5 5 8 2 5 6 6 3 
              index         1 5 1 2 6 ...

Copy operation from the indice array into the data index based on the initial data value 

* https://thrust.github.io/doc/classthrust_1_1permutation__iterator.html

* hmm maybe a tuple<int,int> that contains the data values and indices ? 

Or maybe just use thrust to make the histogram and then apply it with ordinary CUDA ?



EOU
}
thrustexamples-env(){      olocal- ; thrust- ; }
thrustexamples-dir(){  echo $(thrust-sdir)/examples ; }
thrustexamples-bdir(){ echo $(local-base)/env/numerics/thrustexamples ; }
thrustexamples-cd(){  cd $(thrustexamples-dir); }
thrustexamples-get(){
   echo see thrust-
}

thrustexamples-nvcc(){
   local name=${1:-repeated_range}
   local src="$(thrustexamples-dir)/$name.cu" 
   local bin="$(thrustexamples-bdir)/$name"
   mkdir -p $(dirname $bin)
   local cmd="nvcc $src -o $bin"
   echo $cmd
   eval $cmd
   $bin
}


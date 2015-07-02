
template <typename T>
void sparse_histogram_imp(const thrust::device_vector<T>& input,
                                unsigned int input_offset,
                                unsigned int input_itemsize,
                                thrust::device_vector<T>& histogram_values,
                                thrust::device_vector<int>& histogram_counts);


template <typename T, typename S>
void apply_histogram_imp(const thrust::device_vector<T>& sequence,
                               unsigned int sequence_offset,
                               unsigned int sequence_itemsize,
                               thrust::device_vector<S>& target,
                               unsigned int target_offset,
                               unsigned int target_itemsize);


template <typename S>
void strided_copyback( unsigned int n, 
            thrust::host_vector<S>& dest, 
            thrust::device_vector<S>& src, 
            unsigned int src_offset, 
            unsigned int src_itemsize );


template <typename T>
void direct_dump(T* devptr, unsigned int numElements);

const int dev_lookup_n = 32; 

template <typename T>
void update_dev_lookup(T* data); // data needs to have at least dev_lookup_n elements  



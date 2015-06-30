void sparse_histogram_imp(const thrust::device_vector<unsigned long long>& input,
                                thrust::device_vector<unsigned long long>& histogram_values,
                                thrust::device_vector<int>& histogram_counts,
                                thrust::device_vector<int>& histogram_index);


template <typename T>
void direct_dump(T* devptr, unsigned int numElements);

void sparse_histogram_imp(const thrust::device_vector<unsigned long long>& input,
                                thrust::device_vector<unsigned long long>& histogram_values,
                                thrust::device_vector<int>& histogram_counts,
                                thrust::device_vector<int>& histogram_index);

void apply_histogram_imp(const thrust::device_vector<unsigned long long>& input,
                               thrust::device_vector<unsigned long long>& histogram_values,
                               thrust::device_vector<int>& histogram_counts,
                               thrust::device_vector<int>& histogram_index,
                               thrust::device_vector<unsigned int>& target);

template <typename T>
void direct_dump(T* devptr, unsigned int numElements);

const int dev_lookup_n = 32; 
void update_dev_lookup(unsigned long long* data); // data needs to have at least dev_lookup_n elements  



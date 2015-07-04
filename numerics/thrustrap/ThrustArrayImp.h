
template<typename S> 
void resize_imp(
           thrust::device_vector<S>& dvec,
           unsigned int size
            );

template<typename S>
void repeat_to_imp( 
                    unsigned int stride, 
                    unsigned int repeat, 
                    const thrust::device_vector<S>&  source,
                          thrust::device_vector<S>&  target);





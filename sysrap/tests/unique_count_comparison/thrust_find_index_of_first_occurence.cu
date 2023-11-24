// name=thrust_find_index_of_first_occurence ; nvcc $name.cu -o /tmp/$name && /tmp/$name

#include <thrust/device_vector.h>
#include <stdio.h>

struct less_than_or_eq_zero
{
    __host__ __device__ bool operator() (double x) { return x <= 0.; }
};

int main(void)
{
    int N = 6;

    thrust::device_vector<float> D(N);

    D[0] = 3.;
    D[1] = 2.3;
    D[2] = 1.3;
    D[3] = 0.1;
    D[4] = 3.;
    D[5] = 44.;

    thrust::device_vector<float>::iterator iter1 = D.begin();
    thrust::device_vector<float>::iterator iter2 = thrust::find_if(D.begin(), D.begin() + N, less_than_or_eq_zero());
    int d = thrust::distance(iter1, iter2);

    printf("Index = %i\n",d);  // when there are none returns index one past the end

    getchar();

    return 0;
}

/**



C++ find indices of first occurence of multiple values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://stackoverflow.com/questions/25846235/finding-the-indexes-of-all-occurrences-of-an-element-in-a-vector


#include <algorithm> //find_if

std::vector<int> A{1, 0, 1, 1, 0, 0, 0, 1, 0};
std::vector<int> B;

std::vector<int>::iterator it = A.begin();
while ((it = std::find_if(it, A.end(), [](int x){return x == 0; })) != A.end())
{
    B.push_back(std::distance(A.begin(), it));
    it++;
}


**/

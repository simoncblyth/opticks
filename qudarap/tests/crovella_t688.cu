/**
name=crovella_t688 ; nvcc $name.cu -o /tmp/$name && /tmp/$name

https://stackoverflow.com/questions/37013191/is-it-possible-to-create-a-thrusts-function-predicate-for-structs-using-a-given

**/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <iostream>

struct my_score {

  int id;
  int score;
};

const int dsize = 10;

struct copy_func 
{
  int threshold;

  copy_func(int thr) : threshold(thr) {};

  __host__ __device__  bool operator()(const my_score &x)
  {
      return (x.score > threshold);
  }
};

int main(){

  thrust::host_vector<my_score> h_data(dsize);
  thrust::device_vector<my_score> d_result(dsize);
  int my_threshold = 50;
  for (int i = 0; i < dsize; i++)
  {
       h_data[i].id = i;
       h_data[i].score = i * 10;
   }

  thrust::device_vector<my_score> d_data = h_data;

  int rsize = thrust::copy_if(d_data.begin(), d_data.end(), d_result.begin(), copy_func(my_threshold)) - d_result.begin();

  std::cout << "There were " << rsize << " entries with a score greater than " << my_threshold << std::endl;

  return 0;
}


// https://github.com/thrust/thrust/issues/655

/*

simon:thrust blyth$ nvcc -arch=compute_30 longlongb.cu -o /tmp/longlongb
simon:thrust blyth$ /tmp/longlongb
long long: 1000000 1000000
long: 1000000 1000000
long long: 1000000000 1000000000
long: 1000000000 1000000000
long long: 10000000000 1410065408
long: 10000000000 1410065408
simon:thrust blyth$ 



*/



#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

#include <iomanip>

void tryit(long long int n) {
  // with long long

  std::cout << "tryit " << n << std::endl ;  

  long long int s = thrust::reduce(
        thrust::constant_iterator<long long int>(1LL),
        thrust::constant_iterator<long long int>(1LL)+n);

  std::cout << std::setw(20) << "long long: " 
            << std::setw(20) << n 
            << std::setw(20) << s 
            << std::endl;

  // now with long
  long int n1 = n;
  long int s1 = thrust::reduce(
       thrust::constant_iterator<int>(1),
       thrust::constant_iterator<int>(1)+n1);

  std::cout << std::setw(20) << "long: " 
            << std::setw(20) << n1 
            << std::setw(20) << s1 
            << std::endl;

}
int main() {

    std::cout << "sizeof(char)       " << sizeof(char) << std::endl ; 
    std::cout << "sizeof(short)      " << sizeof(short) << std::endl ; 
    std::cout << "sizeof(int)        " << sizeof(int) << std::endl ; 
    std::cout << "sizeof(long)       " << sizeof(long) << std::endl ; 
    std::cout << "sizeof(long long)  " << sizeof(long long) << std::endl ; 
    std::cout << "sizeof(unsigned long long)  " << sizeof(unsigned long long) << std::endl ; 

    tryit(1000000);
    tryit(1000000000);
    tryit(10000000000);
}

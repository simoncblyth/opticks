#include "strided_range.h"

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



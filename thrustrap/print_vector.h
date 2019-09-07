/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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



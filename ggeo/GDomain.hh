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

#pragma once

#include <string>
#include "GGEO_API_EXPORT.hh"


template <class T>
class GGEO_API GDomain {
  public: 
     static GDomain<T>* GetDefaultDomain() ; 
     static GDomain<T>* MakeDefaultDomain() ; 
     static GDomain<T>* MakeCoarseDomain() ; 
     static GDomain<T>* MakeFineDomain() ; 
     static size_t Length(T low, T high, T step);
  private: 
     static GDomain<T>* fDefaultDomain ; 
  public: 
     GDomain(T low, T high, T step);
     virtual ~GDomain() {}
  public: 
     GDomain<T>* makeInterpolationDomain(T step) const ;
  public: 
     T getLow() const {  return m_low ; }   
     T getHigh() const { return m_high ; }   
     T getStep() const { return m_step ; }
     size_t getLength() const { return m_length ; }
  public: 
     std::string desc() const ; 
     void Summary(const char* msg="GDomain::Summary") const ;
     bool isEqual(GDomain<T>* other) const ; 
     T* getValues() const ;   
     T getValue(unsigned i) const ; 
  private:
     T m_low ; 
     T m_high ; 
     T m_step ; 
     size_t m_length ; 


};




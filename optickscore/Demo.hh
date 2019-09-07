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

#include <vector>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "NConfigurable.hpp"

class OKCORE_API Demo : public NConfigurable  {
   public:
       static const char* A ;
       static const char* B ;
       static const char* C ;
       static const char* PREFIX ;
       const char* getPrefix();
   public:
       Demo();

   public:
     // BCfg binding (unused)
     void configureS(const char* , std::vector<std::string> );
     void configureF(const char*, std::vector<float>  );
     void configureI(const char* , std::vector<int>  );
   public:
     // Configurable
     std::vector<std::string> getTags();
     void set(const char* name, std::string& xyz);
     std::string get(const char* name);
   public:
       float getA();
       float getB();
       float getC();

       void setA(float a);
       void setB(float b);
       void setC(float c);

   private:
       float m_a ; 
       float m_b ; 
       float m_c ; 

};



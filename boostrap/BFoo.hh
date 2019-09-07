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

#include <iostream>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template<typename T> 
void foo(T value)
{
    std::cerr << "BFoo"
              << " value " << value
              << std::endl 
              ;
}

template BRAP_API void foo<int>(int);
template BRAP_API void foo<double>(double);
template BRAP_API void foo<char*>(char*);




////////////////  explicit instanciation in .hh /////////////

class BRAP_API BBar {
   public:
        template <typename T>
        void foo(T value)
        {
            std::cerr << "BBar::foo"
                      << " value " << value
                       << std::endl 
              ;

        }   
};


template BRAP_API void BBar::foo<int>(int);
template BRAP_API void BBar::foo<double>(double);
template BRAP_API void BBar::foo<char*>(char*);


////////////////  explicit instanciation in  .cc /////////////

class BRAP_API BCar {
   public:
        template <typename T>
        void foo(T value);

};

/////////////////////////////////////////////////////////////////////////



#include "BRAP_TAIL.hh"




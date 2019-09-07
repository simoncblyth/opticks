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
template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"

//
// Hmm : would be better to use the GPU derived
//       indices rather than going back to the raw sequence
//       which requires sequence data copied back to host 
//
// BUT: this is handy as a check anyhow
//

class NPY_API SeqNPY {
       static const unsigned N ; 
   public:  
       SeqNPY(NPY<unsigned long long>* sequence); 
       virtual ~SeqNPY();

       void dump(const char* msg="SeqNPY::dump");
       int getCount(unsigned code);
       std::vector<int> getCounts();
  private:
       void init();
       void countPhotons();
  private:
        NPY<unsigned long long>* m_sequence ;  // weak 
        int*                     m_counts ; 
 
};




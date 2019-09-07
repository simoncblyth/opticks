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
#include <boost/lexical_cast.hpp>




void check(const char* value)
{
     float percent = boost::lexical_cast<float>(value)*100.f ;   // express as integer percentage 
 
     std::cout << " value "  << value 
               << " percent " << percent
               << std::endl ; 

     unsigned upercent = percent ;
      
     std::cout << " value "  << value 
               << " percent " << percent
               << " upercent " << upercent
               << std::endl ; 
}





int main()
{
     check("0.99");
     check("0.999");

     return 0 ; 
}


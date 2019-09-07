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


#include "NPY_FLAGS.hh"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>

#include "NSlice.hpp"


NSlice::NSlice(unsigned int low_, unsigned int high_, unsigned int step_) 
    :
    low(low_),
    high(high_),
    step(step_),
    _description(0)
{
}



const char* NSlice::description()
{
    if(!_description)
    { 
        char desc[128];
        snprintf(desc, 128, "NSlice  %5d : %5d : %5d ", low, high, step );
        _description = strdup(desc) ;
    }
    return _description ; 
}


unsigned int NSlice::count()
{
    unsigned int n = 0 ;
    for(unsigned int i=low ; i < high ; i+=step ) n+=1 ; 
    return n ;  
}

bool NSlice::isTail(unsigned index, unsigned window)
{
    int h = high ; 
    int i = index ; 
    int o = i - h ; 
    int w = window ; 
    return o < 0 && -o <= w ; 
}

bool NSlice::isHead(unsigned index, unsigned window)
{
    int l = low ; 
    int i = index ; 
    int o = i - l ; 
    int w = window ; 
    return o >= 0 && o < w ; 
}

bool NSlice::isMargin(unsigned index, unsigned window)
{
    return isTail(index, window) || isHead(index, window) ;
}




NSlice::NSlice(const char* slice, const char* delim)
{
    // defaults
    low = 0 ;
    high = 1 ; 
    step = 1 ; 
    _description = 0 ; 


    std::vector<std::string> elem ;
    boost::split(elem, slice, boost::is_any_of(delim));
    unsigned int size = elem.size();

    if(size > 0) low = boost::lexical_cast<unsigned int>(elem[0]);
    if(size > 1) high = boost::lexical_cast<unsigned int>(elem[1]);
    if(size > 2) step = boost::lexical_cast<unsigned int>(elem[2]);

    if(size == 1) high = low + 1 ;  // only provided low


/*
   // strsep has portability issues
    unsigned int i = 0 ;
    char* str = strdup(slice);   
    char* token;
    while ((token = strsep(&str, delim)))
    { 
       switch(i)
       {
          case 0:  low = atoi(token) ; break ; 
          case 1: high = atoi(token) ; break ; 
          case 2: step = atoi(token) ; break ; 
       }
       i++ ;
    }
    if(i == 1) high = low+1 ;  // when only single int provided
*/




}

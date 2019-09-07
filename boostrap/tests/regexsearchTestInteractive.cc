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

#include "regexsearch.hh"
#include <iomanip>


int main(int argc, char** argv)
{
    const char* default_ptn = "<[^>]*>";
    const char* ptn =  argc > 1 ? argv[1] : default_ptn ;  

    std::cout << "search cin for text matching regex " << ptn << std::endl ;

    boost::regex e(ptn);

    pairs_t pairs ; 
    regexsearch( pairs, std::cin , e );
    dump(pairs, "pairs plucked using regexp");


    return 0 ; 
}

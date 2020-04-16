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

//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include "UseBoost.hh"

//#include <boost/version.hpp>


int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : argv[0] ; 
    //const char* path = UseBoost::concat_path( argc, argv ); 

    std::cout << " argv0 " << argv[0] << std::endl ; 
    UseBoost::dump_file_size(argv[0]);
    std::cout << " path " << path << std::endl ; 
    UseBoost::dump_file_size(path);


    UseBoost::dump_version(); 


/*
    std::cout 
          << "Using Boost "     
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;
*/

    return 0;
}



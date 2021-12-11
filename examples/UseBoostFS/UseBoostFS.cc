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
#include <string>
#include <cstring>

#include <boost/filesystem.hpp>

#include <boost/version.hpp>



#define API  __attribute__ ((visibility ("default")))

struct API UseBoostFS 
{
   //static const char* program_location(); 
   static const char* concat_path( int argc, char** argv );
   static void dump_file_size(const char* path);
   static void dump_version();
   static void test_parent_path(const char* path); 

};



namespace fs = boost::filesystem;


void UseBoostFS::dump_file_size(const char* path)
{
    std::cout << "UseBoostFS::dump_file_size: \"" << path  << "\" " << fs::file_size(path) << '\n';
}


const char* UseBoostFS::concat_path(int argc, char** argv)
{
    fs::path p ;
    for(int i=1 ; i < argc ; i++)
    {
         char* a = argv[i] ; 
         if(a) p /= a ; 
    }

    std::string x = p.string() ;
    return strdup(x.c_str());
}

void UseBoostFS::test_parent_path(const char* path)
{
    fs::path p(path) ;
    fs::path pp(p.parent_path()) ; 
    std::cout 
       << " path " << path 
       << " p " << p.string()
       << " pp " << pp.string()
       << std::endl 
      ;
}




void UseBoostFS::dump_version()
{
   std::cout 
          << "UseBoostFS::dump_version "         
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;
}


int main(int argc, char** argv)
{
 
   const char* path =  argc < 2 ? argv[0] : UseBoostFS::concat_path( argc, argv ); 

   UseBoostFS::dump_file_size(path);
   UseBoostFS::dump_version();
   UseBoostFS::test_parent_path(path);

   return 0;

}


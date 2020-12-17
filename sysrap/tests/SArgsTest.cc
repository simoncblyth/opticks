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

#include <cassert>
#include "SArgs.hh"

#include "OPTICKS_LOG.hh"


void test_argforced(int argc, char** argv)
{
    const char* argforced = "--compute --nopropagate --tracer --nogdmlpath" ;
    //const char* argforced = "--nogdmlpath" ;

    SArgs* sa = new SArgs(argc, argv, argforced );

    std::cout << " sa->argc " << sa->argc << std::endl ; 
    sa->dump();

    assert(sa->hasArg("--compute"));
    assert(sa->hasArg("--nopropagate"));
    assert(sa->hasArg("--tracer"));
    assert(sa->hasArg("--nogdmlpath"));
}


void test_starts_with()
{
    std::string e = "--hello" ;  
    std::string f = "hello" ;  
    assert( SArgs::starts_with(e,"--") == true ) ;
    assert( SArgs::starts_with(f,"--") == false ) ;
}

void test_get_first_arg_ending_with(int argc, char** argv)
{
    const char* extra = "--red --green path/to/geometry.gdmlx --blue another.gdml" ;
    SArgs* sa = new SArgs(argc, argv, extra );
    const char* arg = sa->get_first_arg_ending_with(".gdml", NULL ); 
    LOG(info) << arg ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_starts_with(); 
    test_argforced(argc, argv); 

    //test_get_first_arg_ending_with( argc, argv ); 

    return 0 ; 
}
/*
The below should be deduped:

    SArgsTest  --tracer --compute
    SArgsTest  --tracer --compute --nopropagate   

Deduping only has effect between the argforced "extra" additions
and the ordinary args, it does not dedupe the standard args.

*/

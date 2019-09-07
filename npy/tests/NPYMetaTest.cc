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

// TEST=NPYMetaTest om-t
//
#include "OPTICKS_LOG.hh"
#include "NPYMeta.hpp"


void test_setValue( NPYMeta& ym, int item )
{
    ym.setValue("float", 42.f , item) ;
    ym.setValue("double", 42. , item) ;    
    ym.setValue("int",    42 , item ) ;
    ym.setValue("int2",    2 , item ) ;

    std::string str("42") ; 
    ym.setValue("std::string", str , item ) ;
}

void test_getValue( NPYMeta& ym, const char* dir, int item )
{
    if(!NPYMeta::ExistsMeta(dir, item ))
    {
        LOG(warning) << " no such meta " << dir << " " << item ; 
        return ;  
    }

    float v_float = ym.getValue<float>("float", "-1", item) ; assert( v_float == 42.f ); 
    double v_double = ym.getValue<double>("double", "-1", item) ; assert( v_double == 42. ); 
    int  v_int = ym.getValue<int>("int", "-1", item) ; assert( v_int == 42 ); 
    int  v_int2 = ym.getValue<int>("int2", "-1", item) ; assert( v_int2 == 2 ); 

    std::string str("42") ; 
    std::string  v_str = ym.getValue<std::string>("std::string", "-1", item) ; assert( v_str.compare(str) == 0 ) ; 
}

void test_save( NPYMeta& ym , const char* dir )
{
    LOG(info) << "." ; 
    test_setValue( ym, -1 ); 
    test_setValue( ym, 5 ); 
    ym.save(dir); 
}

void test_load( NPYMeta& ym , const char* dir )
{
    LOG(info) << "." ; 

    ym.load(dir); 
    test_getValue( ym, dir, -1 ); 
    test_getValue( ym, dir, 5 ); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* dir = "$TMP/NPYMetaTest" ; 

    NPYMeta ym ; 
    test_save( ym, dir ); 
    test_load( ym, dir ); 

    return 0 ; 
}


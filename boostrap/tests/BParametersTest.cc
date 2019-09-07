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

// TEST=BParametersTest om-t


#include <cassert>
#include "OPTICKS_LOG.hh"
#include "BParameters.hh"

void test_basic()
{
    BParameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();

}

void test_save_load()
{
    BParameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();


    const char* path = "$TMP/parameters.json" ;
    p.save(path);

    BParameters* q = BParameters::Load(path);
    if(!q) return ; 
    q->dump("q");
}


void test_set()
{
    BParameters p ;
    p.add<std::string>("red", "g");
    p.add<std::string>("green", "g");
    p.add<std::string>("blue", "b");
    p.dump();

    p.set<std::string>("red","r");
    p.set<std::string>("cyan","c");
    p.dump();
}

void test_bool_nonexisting()
{
    BParameters p ;

    bool non = p.get<bool>("NonExisting","0");
    assert(non == false); 
    bool oui = p.get<bool>("NonExisting","1");
    assert(oui == true); 
}
void test_bool()
{
    BParameters a ;
    a.add<bool>("Existing",true);
    bool yes = a.get<bool>("Existing","0");
    assert(yes == true); 

    BParameters b ;
    b.add<bool>("Existing",false);
    bool no1 = b.get<bool>("Existing","0");
    assert(no1 == false); 
    bool no2 = b.get<bool>("Existing","1");
    assert(no2 == false); 


}

void test_default_copy_ctor()
{
    BParameters a ;
    a.add<std::string>("red", "g");
    a.add<std::string>("green", "g");
    a.add<std::string>("blue", "b");
    a.dump("a");

    BParameters b(a) ;
    b.dump("b");
}


void test_append()
{
    BParameters a ;
    a.add<std::string>("red", "g");
    a.add<std::string>("green", "g");
    a.add<std::string>("blue", "b");
    a.dump("bef");


    a.append("red", "extra" );
    a.append("red", "extra2" );
    a.append("cyan", "append-on-non-existing" );
 
    a.dump("aft");

    std::string v = a.get<std::string>("red") ;

    LOG(info) << " a.get(red) " << v ; 
}

void test_addEnvvarsWithPrefix()
{
    BParameters a ;
    const char* pfx = "OPTICKS_" ; 
    a.addEnvvarsWithPrefix(pfx);
    a.dump("test_addEnvvarsWithPrefix");
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    /*
    test_basic();
    test_save_load();
    test_set();
    test_bool_nonexisting();
    test_bool();
    test_default_copy_ctor();
    test_append();
    */

    test_addEnvvarsWithPrefix();

    return 0 ; 
}

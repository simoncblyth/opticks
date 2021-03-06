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

// ggv --itemlist
#include <string>
#include "Map.hh"

#include "Opticks.hh"
#include "GItemList.hh"

#include "OPTICKS_LOG.hh"

#include "GGEO_BODY.hh"



void test_base()
{
    GItemList l("testlist");

    l.add("red");
    l.add("green");
    l.add("blue");
    l.add("cyan");
    l.add("magenta");
    l.add("yellow");
    l.dump();

    l.save("$TMP");

    GItemList* t = GItemList::Load("$TMP", "testlist");
    if(t) t->dump();

    Map<std::string, unsigned int>* m = new Map<std::string, unsigned int>() ;

    m->add("yellow",1);
    m->add("magenta",2);
    m->add("cyan",3);

    m->dump("sort order");

    t->setOrder(m->getMap());
    t->sort();
    t->dump("after sort");
}

void test_reldir()
{
    GItemList l("testlist", "GItemListTest/test_reldir");

    l.add("red");
    l.add("green");
    l.add("blue");
    l.add("cyan");
    l.add("magenta");
    l.add("yellow");
    l.dump();

    l.save("$TMP");
}



void test_replaceFields(Opticks* cache)
{
    GItemList* il = GItemList::Load(cache->getIdPath(), "GPmt", "GPmt/0");
    il->dump();
    il->dumpFields();
    il->replaceField(0, "OUTERMATERIAL", "MineralOil" );
    il->replaceField(1, "OUTERSURFACE", "lvPmtHemiCathodeSensorSurface" );
    il->dumpFields("after replace");
}


void test_makeSlice(Opticks* cache)
{
    GItemList* il = GItemList::Load(cache->getIdPath(), "GPmt", "GPmt/0");
    il->dump();

    if(il->getNumKeys() == 0) 
    {
        LOG(fatal) << "test_makeSlice FAILED TO LOAD PMT" ;
        return ; 
    }

    GItemList* sl = il->make_slice("0:4");
    sl->dump("sliced");
}


void test_getIndicesWithKeyEnding()
{
    std::vector<unsigned> indices ; 
    const char* ending = "SensorSurface" ; 

    GItemList l("testlist", "GItemListTest/test_reldir");

    l.add("red");
    l.add("green");
    l.add("blueSensorSurface");
    l.add("cyan");
    l.add("magenta");
    l.add("yellowSensorSurface");
    l.dump();

    l.getIndicesWithKeyEnding( indices, ending ); 

    assert( indices.size() == 2 ) ; 
    assert( indices[0] == 2 );  
    assert( indices[1] == 5 );  
}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ; 


    Opticks ok(argc, argv);

    /*
    test_base();
    test_replaceFields(&ok);
    test_makeSlice(&ok);
    test_reldir();
    */
    test_getIndicesWithKeyEnding();

    return 0 ;

}

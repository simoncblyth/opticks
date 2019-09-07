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

#include <string>

#include "BMap.hh"

#include "OPTICKS_LOG.hh"
#include <boost/property_tree/ptree.hpp>

#include "BJSONParser.hh"

const char* pathSU = "$TMP/BMapTestSU.json" ; 
const char* pathUS = "$TMP/BMapTestUS.json" ;
const char* pathSUi = "$TMP/BMapTestSU.ini" ; 
const char* pathSSi = "$TMP/BMapTestSS.ini" ; 


void test_saveMapSU()
{
   std::map<std::string, unsigned int> index ;
   index["/prefix/red"] = 1 ; 
   index["/prefix/green"] = 2 ; 
   index["/prefix/blue"] = 3 ; 

   BMap<std::string,unsigned int>::save(&index, pathSU );
   BMap<std::string,unsigned int>::save(&index, pathSUi );
}
void test_loadMapSU()
{
   std::map<std::string, unsigned int> index ;
   BMap<std::string, unsigned int>::load(&index, pathSU );
   BMap<std::string, unsigned int>::dump(&index,"loadMapSU");
}


void test_saveMapUS()
{
   std::map<unsigned int, std::string> index ;
   index[0] = "hello0" ; 
   index[1] = "hello1" ; 
   index[10] = "hello10" ; 
   BMap<unsigned int, std::string>::save(&index, pathUS );
}
void test_loadMapUS()
{
   std::map<unsigned int, std::string> index ;
   BMap<unsigned int, std::string>::load(&index, pathUS );
   BMap<unsigned int, std::string>::dump(&index,"loadMapUS");
}


void test_saveIni()
{
   std::map<std::string, std::string> md ;
   md["red"] = "a" ; 
   md["green"] = "b" ; 
   md["blue"] = "c" ; 

   BMap<std::string, std::string>::dump(&md, "saveIni");
   BMap<std::string, std::string>::save(&md, pathSSi );
}

void test_loadIni()
{
   std::map<std::string, std::string> md ;
   BMap<std::string, std::string>::load(&md, pathSSi );
   BMap<std::string, std::string>::dump(&md, "loadIni");
}


void test_LoadJSONString_() {

    namespace pt = boost::property_tree;

    pt::ptree t;
    std::stringstream pippo("{\"size\":1000,\"reserved\":100,\"list\": {\"122\":1,\"123\":3}}");
    // std::stringstream pippo;
    // pippo << "{\"size\":1000,\"reserved\":100,\"list\": {\"122\":1,\"123\":3}}";
    pt::read_json(pippo,t);
    pt::write_json(pippo,t,false);
    pippo.seekg(0,pippo.beg);
    pt::read_json(pippo,t);

    std::cout << "done. " << std::endl;
}

void test_LoadJSONString()
{
   std::map<std::string, unsigned> md ;

   //const char* json = "{\"hello\":\"world\"}" ; 
   //const char* json = "{}" ; 
   std::map<std::string, int> mattbl;
   mattbl["LS"] = 48;
   mattbl["Acrylic"] = 24;
   std::stringstream ss;

   ss << "{" << std::endl;
   for (std::map<std::string, int>::iterator it = mattbl.begin();
        it != mattbl.end(); ++it) {
       ss << "\"" << it->first << "\"" << ":" << it->second << "," << std::endl;
   }
   ss << "\"" << "ENDMAT" << "\"" << ":999999" << std::endl;
   ss << "}" << std::endl;
   std::string json_string = ss.str();
   const char* json = json_string.c_str() ; 
   std::cout << json << std::endl;

   // namespace pt = boost::property_tree;

   // std::cout << "READ JSON start." << std::endl;

   // pt::ptree t;
   // std::stringstream pippo(json);
   // pt::read_json(pippo,t);
   // std::cout << "READ JSON done." << std::endl;

   BMap<std::string, unsigned>::LoadJSONString(&md, json, 0 );
   BMap<std::string, unsigned>::dump(&md, "LoadJSONString");

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    LOG(info) << argv[0] ;

/*  
    test_saveMapSU();
    test_loadMapSU();

    test_saveMapUS();
    test_loadMapUS();

    test_saveIni();
    test_loadIni();

*/

    test_LoadJSONString_();
    test_LoadJSONString();

    return 0 ; 
}


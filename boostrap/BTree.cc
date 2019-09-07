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

#include "BFile.hh"
#include "BTree.hh"

#include <string>
#include <sstream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "BJSONParser.hh"

#include <boost/filesystem.hpp>


#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

/**

https://stackoverflow.com/questions/10260688/boostproperty-treejson-parser-and-two-byte-wide-characters

find that this is escaping slashes in the output json
and there seems no way to avoid this

https://www.boost.org/doc/libs/1_70_0/boost/property_tree/json_parser/detail/write.hpp


**/

void BTree::saveTree(const pt::ptree& t , const char* path_)
{
    std::string path = BFile::preparePath(path_, true);
    LOG(verbose) << "BTree::saveTree "
               << " path_ " << path_ 
               << " path " << path 
               ; 


    fs::path fpath(path);
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::write_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::write_ini(path, t );
    else
        LOG(warning) << "saveTree cannot write to path with extension " << ext ; 
}


int BTree::loadTree(pt::ptree& t , const char* path)
{
    fs::path fpath(path);
    LOG(debug) << "BTree.loadTree: "
              << " load path: " << path;

    if (!(fs::exists(fpath ) && fs::is_regular_file(fpath))) 
    {
        LOG(error)
             << "can't find file " << path;

        //assert(0); 
        return 1;
    }
    std::string ext = fpath.extension().string();
    if(ext.compare(".json")==0)
        pt::read_json(path, t );
    else if(ext.compare(".ini")==0)
        pt::read_ini(path, t );
    else
        LOG(warning) << "readTree cannot read path with extension " << ext ; 

    return 0 ; 
}

int BTree::loadJSONString(pt::ptree& t , const char* json)
{
    std::stringstream ss ; 
    ss << json ; 
    // ss.seekg(0, ss.beg);

    pt::read_json(ss, t );
       
    return 0 ; 
}



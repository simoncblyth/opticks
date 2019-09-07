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


#include "OPTICKS_LOG.hh"
#include "BFile.hh"

#include "YOGMaker.hh"
#include "YOG.hh"
#include "YOGGeometry.hh"

using YOG::Geometry ; 
using YOG::Sc ; 
using YOG::Maker ; 

void test_demo_geom()
{
    Geometry geom(3) ; 
    geom.make_triangle();

    Sc* sc = new Sc(0) ;  
    Maker ym(sc) ; 
    ym.demo_create(geom); 

    std::string path = BFile::FormPath("$TMP/yoctoglrap/tests/YOGMakerTest/YOGMakerTest.gltf");

    ym.convert();
    ym.save(path.c_str());
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_demo_geom();

    return 0 ; 
}



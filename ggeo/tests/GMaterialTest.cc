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


#include "GProperty.hh"
#include "GMaterial.hh"

#include "OPTICKS_LOG.hh"


struct GMaterialTest
{
    static void test_zero()
    {
        LOG(info); 
        GMaterial* mat = new GMaterial("test", 0);
        mat->Summary(); 
    }

    static void test_addProperty()
    {
        LOG(info); 
        GMaterial* mat = new GMaterial("demo", 0);

        double domain[]={1.,2.,3.,4.,5.,6.,7.};
        double vals[]  ={10.,20.,30.,40.,50.,60.,70.};

        mat->addPropertyStandardized("pname", vals, domain, sizeof(domain)/sizeof(domain[0]) );

        GProperty<double>* prop = mat->getProperty("pname");
        prop->Summary("prop dump");
    }
}; 



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    GMaterialTest::test_zero();
    GMaterialTest::test_addProperty();


    return 0 ;
}


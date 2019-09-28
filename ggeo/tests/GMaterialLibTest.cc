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

// op --mat
//  TEST=GMaterialLibTest om-t 

#include <string>
#include <iostream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>


#include "SSys.hh"
#include "SAbbrev.hh"

#include "BFile.hh"
#include "BTxt.hh"
#include "BStr.hh"

#include "NMeta.hpp"

#include "Opticks.hh"
#include "OpticksAttrSeq.hh"

#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GDomain.hh"



#include "OPTICKS_LOG.hh"


void colorcodes(GMaterialLib* mlib)
{
    std::vector<unsigned int> cc = mlib->getAttrNames()->getColorCodes();

    //std::copy( cc.begin(), cc.end(), std::ostream_iterator<unsigned int>(std::cout, "\n"));

    for(unsigned int i=0 ; i < cc.size() ; i++)
    {
       std::cout << std::setw(5) << i 
                 << std::setw(10) << std::hex << cc[i] << std::dec
                 << std::endl ; 
    }
}


void attrdump( GMaterialLib* mlib)
{
    const char* mats = "Acrylic,GdDopedLS,LiquidScintillator,ESR,MineralOil" ;

    OpticksAttrSeq* amat = mlib->getAttrNames();

    amat->dump(mats);
}


void test_getMaterial(GMaterialLib* mlib)
{
    const char* name = "Water" ;
    GMaterial* mat = mlib->getMaterial(name);
    mlib->dump(mat, name);
}


void test_addTestMaterial(GMaterialLib* mlib)
{
    // see GGeo::addTestMaterials

    GProperty<float>* f2 = GProperty<float>::load("$OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/glass/schott/F2.npy");
    if(f2)
    {
        f2->Summary("F2 ri", 100);
        GMaterial* raw = new GMaterial("GlassSchottF2", mlib->getNumMaterials() );
        raw->addPropertyStandardized( GMaterialLib::refractive_index_local, f2 ); 

        mlib->setClosed(false);  // OK for testing only 
        mlib->add(raw);
    }
}


void test_setMaterialPropertyValues(GMaterialLib* mlib)
{
    // this can no be done with --noreem option
    mlib->setMaterialPropertyValues("GdDopedLS", "reemission_prob", 0.f );
    mlib->setMaterialPropertyValues("LiquidScintillator", "reemission_prob", 0.f );
}



void test_interpolatingCopyCtor(GMaterialLib* mlib)
{
    GDomain<float>* idom = mlib->getStandardDomain()->makeInterpolationDomain(1.f);
    GMaterialLib* ilib = new GMaterialLib(mlib, idom);

    ilib->dump("test_interpolatingCopyCtor (ilib)");

}

void test_getLocalKey(GMaterialLib* mlib)
{
    const char* keys_ = "refractive_index absorption_length scattering_length reemission_prob detect non_existing_key absorb" ; 
    std::vector<std::string> keys ; 
    BStr::split(keys, keys_, ' '); 

    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const char* key = keys[i].c_str(); 
        const char* lkey = mlib->getLocalKey(key); 
        LOG(info) 
            << " key " << std::setw(30) << key 
            << " lkey " << std::setw(30) << lkey
            ;  
    }
}




void test_load(int argc, char** argv)
{

    SSys::setenvvar("IDPATH", "$TMP", true );

    Opticks ok(argc, argv);
    ok.configure();

    LOG(info) << " ok " ; 

    GMaterialLib* mlib = GMaterialLib::load(&ok);

    mlib->dumpDomain();  

    LOG(info) << " after load " ; 
    test_addTestMaterial(mlib);
    //test_setMaterialPropertyValues(mlib);

    mlib->dump("dump");
    LOG(info) << " after dump " ; 


    //test_interpolatingCopyCtor(mlib);
    test_getLocalKey(mlib) ; 
}







void test_createAbbrevMeta()
{
    std::string path = BFile::FormPath("$IDPATH/GItemList/GMaterialLib.txt");
    LOG(info) << path ;

    BTxt bt(path.c_str());
    bt.read();
    bt.dump("test_createAbbrevMeta");

    NMeta* abbrevmeta = GPropertyLib::CreateAbbrevMeta( bt.getLines() );
    abbrevmeta->dump(); 

    NMeta* libmeta = new NMeta ; 
    libmeta->setObj("abbrev", abbrevmeta );  

    libmeta->dump() ; 

    libmeta->save("$TMP/ggeo/GMaterialLib/GPropertyLibMetadata.json");
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_load(argc, argv); 
    test_createAbbrevMeta(); 

    return 0 ;
}


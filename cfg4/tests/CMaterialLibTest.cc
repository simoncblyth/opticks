// op --cmat 
// op --cmat 0
// op --cmat GdDopedLS

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "OpticksMode.hh"
#include "CMaterialLib.hh"
#include "CMPT.hh"
#include "CVec.hh"
#include "CFG4_BODY.hh"
#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"

#include "PLOG.hh"


void test_CMPT(CMaterialLib* clib)
{
    const char* shortname = "Acrylic" ; 
    const CMPT* mpt = clib->getG4MPT(shortname);
    mpt->dump(shortname);

    const char* lkey = "GROUPVEL" ; 

    CVec* vec = mpt->getCVec(lkey);
    vec->dump(lkey);
}

void test_MaterialValueMap(CMaterialLib* clib)
{

    std::map<std::string, float> vmp ; 
    const char* matnames = "GdDopedLS,Acrylic,LiquidScintillator,Acrylic,MineralOil" ; 
    const char* lkey = "GROUPVEL" ; 
    float nm = 430.f ; 

   //  now done standardly in postconvert


    clib->fillMaterialValueMap(vmp, matnames, lkey, nm);
    CMaterialLib::dumpMaterialValueMap(matnames, vmp);

    std::string x_empty = CMaterialLib::firstKeyForValue( 200.f , vmp, 0.001f); 
    assert(x_empty.empty());

    const char* Acrylic = "Acrylic" ; 
    const char* GdDopedLS = "GdDopedLS" ; 
    const char* LiquidScintillator = "LiquidScintillator" ; 
    const char* MineralOil = "MineralOil" ; 

    std::string x_acrylic = CMaterialLib::firstKeyForValue( 192.811f , vmp, 0.001f); 
    assert(strcmp(x_acrylic.c_str(),Acrylic)==0);

    std::string x_ls = CMaterialLib::firstKeyForValue( 194.539f , vmp, 0.001f); 
    assert(strcmp(x_ls.c_str(),GdDopedLS)==0 || strcmp(x_ls.c_str(),LiquidScintillator)==0 );

    std::string x_mo = CMaterialLib::firstKeyForValue( 197.149f , vmp, 0.001f); 
    assert(strcmp(x_mo.c_str(),MineralOil)==0);

}

void test_GroupvelLookup_failing(CMaterialLib* clib)
{
    // this functionality was only used for debugging, i recall ??
    // now failing due to change to finer sampling ?

    float groupvel =  197.149f ;
    std::string mat = clib->firstMaterialWithGroupvelAt430nm( groupvel, 0.001f );
    LOG(info) << "lookup by groupvel value " << groupvel << " yields mat " << mat ; 

    assert(strcmp(mat.c_str(),"MineralOil")==0);


/*

2017-06-15 15:07:41.487 INFO  [7648214] [CMaterialLib::dumpMaterial@339] CMaterialLib::dump name MineralOil
GPropertyMap<T>::make_table vprops 4 cprops 1 dprops 0 eprops 0 fprops 0 gprops 0
              domain           ABSLENGTH            GROUPVEL            RAYLEIGH              RINDEX
                 520             22980.5             200.106             81100.8              1.4661
                 500             23628.8               199.8             68424.9             1.46734
                 480               24277              198.79             58710.2             1.46876
                 460             24925.3             198.053             52038.9             1.47063
                 440             25254.7             197.783             45367.7             1.47251
                 420             24706.1             196.485             36028.8             1.47458
                 400             11655.2             192.606             27963.7             1.47743
                 380             4941.75             189.538             23891.6             1.48264
                 360              1078.9             189.538             19819.4             1.48786
*/

   /*
     OLD:
           Acrylic               192.811
           GdDopedLS             194.539
  LiquidScintillator             194.539
          MineralOil             197.149

    */
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    GGEO_LOG__ ;  
    CFG4_LOG__ ;  

    LOG(info) << argv[0] ; 

    Opticks ok(argc, argv);
    
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4

    OpticksHub hub(&ok); 

    CMaterialLib* clib = new CMaterialLib(&hub); 

    LOG(info) << argv[0] << " convert " ; 

    clib->convert();

    LOG(info) << argv[0] << " dump " ; 

    clib->dump();

    //test_GroupvelLookup_failing(clib);

    return 0 ; 
}





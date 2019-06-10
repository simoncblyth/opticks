/**

   VERBOSE=1 NSensorListTest

simon:opticks blyth$ NSensorListTest 6871 
2017-11-30 15:47:38.683 ERROR [640228] [NSensorList::load@77] NSensorList::load 
 idmpath:   /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.idmap
nodeIndex 6871 sensor NSensor  index   2390 idhex 1051306 iddec 17109766 node_index   6871 name /dd/Geometry/Pool/lvNearPoolIWS#pvVetoPmtNearInn#pvNearInnWall3#pvNearInnWall3:6#pvVetoPmtUnit#pvPmtMount#pvMountRib1s#pvMountRib1s:1#pvMountRib1unit NOT-CATHODE 
 simon:opticks blyth$ 



**/

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "BOpticksResource.hh"

#include "NSensorList.hpp"
#include "NSensor.hpp"


struct NSensorListTest
{
    NSensorListTest( const char* idpath )
        :
        _testgeo(false),
        _res(_testgeo)
    {
        _res.setupViaID(idpath); 
        const char* idmpath = _res.getIdMapPath(); 
        assert( idmpath ); 
        _sens.load(idmpath);
    }
   
    bool             _testgeo ; 
    BOpticksResource _res ; 
    NSensorList      _sens ; 
};




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* idpath = SSys::getenvvar("IDPATH");
    if(!idpath) return 0 ; 

    NSensorListTest slt(idpath);

    if(getenv("VERBOSE")) slt._sens.dump();

    for(int i=1 ; i < argc ; i++)
    {
       unsigned int nodeIndex = atoi(argv[i]) ;
       NSensor* sensor = slt._sens.findSensorForNode(nodeIndex);
       printf("nodeIndex %u sensor %s \n ", nodeIndex, ( sensor ? sensor->description().c_str() : "NULL" ) );
    }


    return 0 ;
}

/*


simon:cu blyth$ ggv --sensor 3198 3199 3200 3201 3202 3203 3204 
[2015-10-07 16:12:20.611580] [0x000007fff7448031] [debug]   GSensorList::load 
 idpath:   /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
 pdir:     /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300
 filename: g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
 daepath:  /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae
 idmpath:  /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap
[2015-10-07 16:12:20.896478] [0x000007fff7448031] [info]    GSensorList::read  path /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.idmap desc GSensorList:  GSensor count 6888 distinct identier count 684
nodeIndex 0 sensor NULL 
 nodeIndex 3198 sensor NULL 
 nodeIndex 3199 sensor GSensor  index      0 idhex 1010101 iddec 16843009 node_index   3199 name /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt 
 nodeIndex 3200 sensor GSensor  index      1 idhex 1010101 iddec 16843009 node_index   3200 name /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum 
 nodeIndex 3201 sensor GSensor  index      2 idhex 1010101 iddec 16843009 node_index   3201 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode 
 nodeIndex 3202 sensor GSensor  index      3 idhex 1010101 iddec 16843009 node_index   3202 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom 
 nodeIndex 3203 sensor GSensor  index      4 idhex 1010101 iddec 16843009 node_index   3203 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode 
 nodeIndex 3204 sensor NULL 
 simon:cu blyth$ 



*/

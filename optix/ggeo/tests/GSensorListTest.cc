#include "GSensorList.hh"
#include "GSensor.hh"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    if(!idpath) printf("%s : requires IDPATH envvar \n", argv[0]);

    GSensorList sens;
    sens.load(idpath, "idmap");

    if(getenv("VERBOSE")) sens.dump();

    for(unsigned int i=1 ; i < argc ; i++)
    {
       unsigned int nodeIndex = atoi(argv[i]) ;
       GSensor* sensor = sens.findSensorForNode(nodeIndex);
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

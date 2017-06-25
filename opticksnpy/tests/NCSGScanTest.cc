/**
Scans individual CSG trees::

    delta:opticksnpy blyth$ SCAN=0,0,1500,0,0,1,0,200,5 NCSGScanTest /tmp/blyth/opticks/tgltf/extras/22
    BStr::fsplitEnv envvar SCAN line 0,0,1500,0,0,1,0,200,5 fallback 0,0,128,0,0,1,-1,1,0.001 elem.size 9
    2017-06-25 09:32:18.920 INFO  [854818] [NScan::scan@62] NScan::scan
     origin {    0.0000    0.0000 1500.0000} direction {    0.0000    0.0000    1.0000} range {    0.0000  200.0000    5.0000}
     t     0.0000 x     0.0000 y     0.0000 z  1500.0000 :   -35.0000 l   -35.0000 r    35.0000
     t     5.0000 x     0.0000 y     0.0000 z  1505.0000 :   -30.0000 l   -30.0000 r    30.0000
     t    10.0000 x     0.0000 y     0.0000 z  1510.0000 :   -25.0000 l   -25.0000 r    25.0000
     t    15.0000 x     0.0000 y     0.0000 z  1515.0000 :   -20.0000 l   -20.0000 r    20.0000        INSIDE L (base cylinder)
     t    20.0000 x     0.0000 y     0.0000 z  1520.0000 :   -15.0000 l   -15.0000 r    15.0000
     t    25.0000 x     0.0000 y     0.0000 z  1525.0000 :   -10.0000 l   -10.0000 r    10.0000
     t    30.0000 x     0.0000 y     0.0000 z  1530.0000 :    -5.0000 l    -5.0000 r     5.0000
     t    35.0000 x     0.0000 y     0.0000 z  1535.0000 :     0.0000 l     0.0000 r    -0.0000    <--- INTERNAL ISO ZERO
     t    40.0000 x     0.0000 y     0.0000 z  1540.0000 :    -5.0000 l     5.0000 r    -5.0000
     t    45.0000 x     0.0000 y     0.0000 z  1545.0000 :   -10.0000 l    10.0000 r   -10.0000
     t    50.0000 x     0.0000 y     0.0000 z  1550.0000 :   -15.0000 l    15.0000 r   -15.0000
     t    55.0000 x     0.0000 y     0.0000 z  1555.0000 :   -20.0000 l    20.0000 r   -20.0000        INSIDE R (top cone) 
     t    60.0000 x     0.0000 y     0.0000 z  1560.0000 :   -25.0000 l    25.0000 r   -25.0000 
     t    65.0000 x     0.0000 y     0.0000 z  1565.0000 :   -30.0000 l    30.0000 r   -30.0000
     t    70.0000 x     0.0000 y     0.0000 z  1570.0000 :   -35.0000 l    35.0000 r   -35.0000
     t    75.0000 x     0.0000 y     0.0000 z  1575.0000 :   -35.7292 l    40.0000 r   -35.7292
     t    80.0000 x     0.0000 y     0.0000 z  1580.0000 :   -30.7292 l    45.0000 r   -30.7292
     t    85.0000 x     0.0000 y     0.0000 z  1585.0000 :   -25.7292 l    50.0000 r   -25.7292
     t    90.0000 x     0.0000 y     0.0000 z  1590.0000 :   -20.7292 l    55.0000 r   -20.7292
     t    95.0000 x     0.0000 y     0.0000 z  1595.0000 :   -15.7292 l    60.0000 r   -15.7292
     t   100.0000 x     0.0000 y     0.0000 z  1600.0000 :   -10.7292 l    65.0000 r   -10.7292
     t   105.0000 x     0.0000 y     0.0000 z  1605.0000 :    -5.7292 l    70.0000 r    -5.7292
     t   110.0000 x     0.0000 y     0.0000 z  1610.0000 :    -0.7292 l    75.0000 r    -0.7292       <--- possible other internal zero inside here 
     t   115.0000 x     0.0000 y     0.0000 z  1615.0000 :    -4.2708 l    80.0000 r    -4.2708
     t   120.0000 x     0.0000 y     0.0000 z  1620.0000 :    -4.4397 l    85.0000 r    -4.4397     ----------------------
     t   125.0000 x     0.0000 y     0.0000 z  1625.0000 :     0.5603 l    90.0000 r     0.5603
     t   130.0000 x     0.0000 y     0.0000 z  1630.0000 :     5.5603 l    95.0000 r     5.5603       OUTSIDE R 
     t   135.0000 x     0.0000 y     0.0000 z  1635.0000 :    10.5603 l   100.0000 r    10.5603
     t   140.0000 x     0.0000 y     0.0000 z  1640.0000 :    15.5603 l   105.0000 r    15.5603
     t   145.0000 x     0.0000 y     0.0000 z  1645.0000 :    20.5603 l   110.0000 r    20.5603
     t   150.0000 x     0.0000 y     0.0000 z  1650.0000 :    25.5603 l   115.0000 r    25.5603
     t   155.0000 x     0.0000 y     0.0000 z  1655.0000 :    30.5603 l   120.0000 r    30.5603


**/

#include <iostream>

#include "BFile.hh"
#include "BStr.hh"


#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NScan.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

void ManualScan(NCSG* csg)
{
    const char* fallback = "0,0,128,0,0,1,-1,1,0.001" ; 
    std::vector<float> f ; 
    BStr::fsplitEnv(f, "SCAN", fallback, ',' );

    bool has9 = f.size() == 9 ;
    if(!has9) LOG(fatal) << "NCSGScan"
                         << " SCAN envvar required 9 comma delimited elements" 
                         << " got " << f.size()
                        ;
    assert(has9);
    nnode* root = csg->getRoot();
    glm::vec3 origin(    f[0],f[1],f[2] );
    glm::vec3 direction( f[3],f[4],f[5] );
    glm::vec3 range(     f[6],f[7],f[8] );

    unsigned verbosity = 2 ; 
    std::vector<float> sd ; 
    NScan scan(*root, verbosity);
    scan.scan(sd, origin, direction, range);
}

void AutoScan(NCSG* csg)
{
    unsigned verbosity = 2 ; 
    nnode* root = csg->getRoot();
    NScan scan(*root, verbosity);
    scan.autoscan();
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    NCSG* csg = NCSG::LoadCSG( argc > 1 ? argv[1] : NULL );
    if(!csg) return 0 ; 

    if(BStr::existsEnv("SCAN"))
    {
        ManualScan(csg);
    }
    else
    {
        AutoScan(csg);
    }
    return 0 ; 
}



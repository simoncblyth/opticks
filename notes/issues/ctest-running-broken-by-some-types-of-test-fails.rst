ctest-running-broken-by-some-types-of-test-fails
=====================================================

Issue
----------

Notice mixed dates on the subproj ctest logs and unexpected missing fails, the
below is from opticks-t run on silver over ssh from gold.  



::


    LOGS:
    CTestLog :                 okop :      1/     5 : 2019-09-21 23:54:33.347527 : /home/blyth/local/opticks/build/okop/ctest.log 
    CTestLog :               oglrap :      0/     2 : 2019-09-21 23:54:33.823529 : /home/blyth/local/opticks/build/oglrap/ctest.log 
    CTestLog :            opticksgl :      0/     0 : 2019-09-21 23:54:34.166530 : /home/blyth/local/opticks/build/opticksgl/ctest.log 
    CTestLog :                   ok :      1/     5 : 2019-09-21 23:55:26.981726 : /home/blyth/local/opticks/build/ok/ctest.log 
    CTestLog :                extg4 :      0/    18 : 2019-09-21 23:55:33.947752 : /home/blyth/local/opticks/build/extg4/ctest.log 
    CTestLog :                 cfg4 :      0/    34 : 2019-09-22 00:16:07.470336 : /home/blyth/local/opticks/build/cfg4/ctest.log 
    CTestLog :                 okg4 :      1/     1 : 2019-09-22 00:17:14.404585 : /home/blyth/local/opticks/build/okg4/ctest.log 
    CTestLog :                 g4ok :      0/     1 : 2019-09-22 00:17:14.928587 : /home/blyth/local/opticks/build/g4ok/ctest.log 
    CTestLog :                  ana :      0/     1 : 2019-09-22 00:17:22.542615 : /home/blyth/local/opticks/build/ana/ctest.log 
    CTestLog :             analytic :      0/     1 : 2019-09-22 00:17:22.912616 : /home/blyth/local/opticks/build/analytic/ctest.log 
    CTestLog :                  bin :      0/     1 : 2019-09-22 00:17:23.274618 : /home/blyth/local/opticks/build/bin/ctest.log 
    CTestLog :          integration :      1/     2 : 2019-09-22 11:24:11.665061 : /home/blyth/local/opticks/build/integration/ctest.log 
    CTestLog :               okconf :      0/     1 : 2019-09-22 12:41:17.004251 : /home/blyth/local/opticks/build/okconf/ctest.log 
    CTestLog :               sysrap :      0/    42 : 2019-09-22 12:41:17.913255 : /home/blyth/local/opticks/build/sysrap/ctest.log 
    CTestLog :             boostrap :      0/    38 : 2019-09-22 12:41:20.007264 : /home/blyth/local/opticks/build/boostrap/ctest.log 
    CTestLog :                  npy :      0/   121 : 2019-09-22 12:41:26.927295 : /home/blyth/local/opticks/build/npy/ctest.log 
    CTestLog :           yoctoglrap :      0/     4 : 2019-09-22 12:41:27.386297 : /home/blyth/local/opticks/build/yoctoglrap/ctest.log 
    CTestLog :          optickscore :      0/    30 : 2019-09-22 12:41:28.311301 : /home/blyth/local/opticks/build/optickscore/ctest.log 
    CTestLog :                 ggeo :      0/    53 : 2019-09-22 12:41:42.899366 : /home/blyth/local/opticks/build/ggeo/ctest.log 
    CTestLog :            assimprap :      0/     3 : 2019-09-22 12:41:43.338368 : /home/blyth/local/opticks/build/assimprap/ctest.log 
    CTestLog :          openmeshrap :      0/     1 : 2019-09-22 12:41:43.737370 : /home/blyth/local/opticks/build/openmeshrap/ctest.log 
    CTestLog :           opticksgeo :      0/     3 : 2019-09-22 12:41:45.092376 : /home/blyth/local/opticks/build/opticksgeo/ctest.log 
    CTestLog :              cudarap :      0/     7 : 2019-09-22 12:41:46.794384 : /home/blyth/local/opticks/build/cudarap/ctest.log 
    CTestLog :            thrustrap :      0/    17 : 2019-09-22 12:41:56.684428 : /home/blyth/local/opticks/build/thrustrap/ctest.log 
    CTestLog :             optixrap :      1/    25 : 2019-09-22 12:42:20.324534 : /home/blyth/local/opticks/build/optixrap/ctest.log 


    FAILS:  5   / 416   :  Sun Sep 22 12:42:20 2019   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     42.69  
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     45.37  
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     66.55  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.55   
      18 /25  Test #18 : OptiXRapTest.interpolationTest                ***Failed                      5.95   
    (base) [blyth@gilda03 opticks]$ 


Last FAIL from optixrap::


    2019-09-22 12:42:15.576 INFO  [126088] [OLaunchTest::launch@80] OLaunchTest entry   0 width     761 height      31 ptx                               interpolationTest.cu prog                                  interpolationTest
    2019-09-22 12:42:16.915 INFO  [126088] [interpolationTest::launch@158] OLaunchTest entry   0 width     761 height      31 ptx                               interpolationTest.cu prog                                  interpolationTest
    2019-09-22 12:42:16.919 INFO  [126088] [interpolationTest::launch@165]  save  base $TMP/interpolationTest name interpolationTest_interpol.npy
    which: no interpolationTest_interpol.py in (/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/bin:/home/blyth/env/bin:/home/blyth/anaconda2/bin:/home/blyth/anaconda2/condabin:/home/blyth/opticks/bin:/home/blyth/opticks/ana:/home/blyth/anaconda2/bin:/home/blyth/.cargo/bin:/home/blyth/local/opticks/lib:/home/blyth/local/bin:/usr/local/cuda-10.1/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:/home/blyth/.local/bin:/home/blyth/bin)
    2019-09-22 12:42:16.970 INFO  [126088] [interpolationTest::ana@179]  m_script interpolationTest_interpol.py path 
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NameError: name 'okop' is not defined
    2019-09-22 12:42:17.042 INFO  [126088] [SSys::run@91] python  rc_raw : 256 rc : 1
    2019-09-22 12:42:17.042 ERROR [126088] [SSys::run@98] FAILED with  cmd python  RC 1
    2019-09-22 12:42:17.042 INFO  [126088] [interpolationTest::ana@185]  RC 1

          Start 19: OptiXRapTest.ORngTest
    19/25 Test #19: OptiXRapTest.ORngTest .......................................   Passed    1.41 sec
          Start 20: OptiXRapTest.intersectAnalyticTest
    20/25 Test #20: OptiXRapTest.intersectAnalyticTest ..........................   Passed    0.04 sec
          Start 21: OptiXRapTest.intersectAnalyticTest.iaDummyTest


Has::

   NameError: name 'okop' is not defined


okop is the sub after optixrap::

    [blyth@localhost sysrap]$ om-subs
    okconf
    sysrap
    boostrap
    npy
    yoctoglrap
    optickscore
    ggeo
    assimprap
    openmeshrap
    opticksgeo
    cudarap
    thrustrap
    optixrap
    okop
    oglrap
    opticksgl
    ok
    extg4
    cfg4
    okg4
    g4ok
    integration
    ana
    analytic
    bin



Which points finger at::

   SSys::POpen 

::

    108 /**
    109 SSys::POpen
    110 -------------
    111 Run command and get output into string, 
    112 Newlines are removed when chomp is true.
    113 
    114 **/
    115 
    116 std::string SSys::POpen(const char* cmd, bool chomp)
    117 {
    118     LOG(info) << "[ " << cmd ; 
    119     
    120     std::stringstream ss ; 
    121     FILE *fp = popen(cmd, "r");
    122     char line[512];    
    123     while (fgets(line, sizeof(line), fp) != NULL)
    124     {
    125        if(chomp) line[strcspn(line, "\n")] = 0;
    126        //LOG(info) << "[" << line << "]" ; 
    127        ss << line ;
    128     }
    129     pclose(fp);
    130     LOG(info) << "] " << cmd ;
    131     return ss.str();
    132 }


Somehow some stray string gets passed to python ?


::

    (base) [blyth@gilda03 opticks]$ ini
    (base) [blyth@gilda03 opticks]$ which interpolationTest_interpol.py
    ~/local/opticks/bin/interpolationTest_interpol.py
    (base) [blyth@gilda03 opticks]$ 



Found cause, the below needs to error check the result of which 
otherwise will pass random error messages to python::

    175 int interpolationTest::ana()
    176 {
    177     bool chomp = true ;
    178     std::string path = SSys::POpen("which", m_script, chomp);
    179     LOG(info)
    180          << " m_script " << m_script
    181          << " path " << path
    182          ;
    183 
    184     int RC = SSys::exec("python",path.c_str());
    185     LOG(info) << " RC " << RC ;
    186     return RC ;
    187 }




opticks-t-18-of-371-fails-on-epsilon
=====================================

::

    FAILS:
      73 /120 Test #73 : NPYTest._RecordsNPYTest                       ***Exception: SegFault         0.01   
      75 /120 Test #75 : NPYTest.PhotonsNPYTest                        ***Exception: SegFault         0.01   
      2  /24  Test #2  : OpticksCoreTest.IndexerTest                   ***Exception: SegFault         0.01   
      6  /24  Test #6  : OpticksCoreTest.OpticksEventSpecTest          ***Exception: SegFault         0.01   
      25 /50  Test #25 : GGeoTest.GItemIndexTest                       ***Exception: SegFault         0.01   
      2  /15  Test #2  : ThrustRapTest.TBufTest                        ***Exception: SegFault         0.79   

      1  /33  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.29   
      2  /33  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.29   
      3  /33  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.97   
      5  /33  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.86   
      6  /33  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.81   
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.90   
      23 /33  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.88   
      25 /33  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.28   
      28 /33  Test #28 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.89   
      31 /33  Test #31 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.29   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.29   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.90   
    epsilon:build blyth$ 



After fix the BResource issue::

    totals  13  / 372 


    FAILS:
      14 /18  Test #14 : OptiXRapTest.OEventTest                       ***Exception: Child aborted    0.28   
      1  /33  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.32   
      2  /33  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.29   
      3  /33  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.88   
      5  /33  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.83   
      6  /33  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.81   
      7  /33  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.89   
      23 /33  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.86   
      25 /33  Test #25 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.28   
      28 /33  Test #28 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.88   
      31 /33  Test #31 : CFG4Test.CCerenkovGeneratorTest               ***Exception: Child aborted    0.29   
      32 /33  Test #32 : CFG4Test.CGenstepSourceTest                   ***Exception: Child aborted    0.30   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.90   
    epsilon:build blyth$ 




OPTICKS_EVENT_BASE issue : added BFileTest to reproduce at lower level
----------------------------------------------------------------------------------

\_RecordsNPYTest::

    (lldb) p ubase
    (std::__1::string) $0 = "$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1"
    (lldb) f 4
    frame #4: 0x000000010088de1f libBoostRap.dylib`BFile::FormPath(path="$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1", sub=0x0000000000000000, name=0x0000000000000000, extra1=0x0000000000000000, extra2=0x0000000000000000) at BFile.cc:399
       396 	
       397 	   if(path[0] == '$')
       398 	   {
    -> 399 	      xpath.assign(expandvar(path));
       400 	   } 
       401 	   else if(path[0] == '~')
       402 	   {
    (lldb) 



::

    121            else if(evalue.compare("OPTICKS_EVENT_BASE")==0)
    122            {
    123                const char* evtbase = BResource::Get("evtbase") ;
    124                if( evtbase != NULL )
    125                {
    126                    evalue = evtbase ;
    127                }
    128                else
    129                {
    130                    evalue = BResource::Get("tmpuser_dir") ;
    131                    //evalue = usertmpdir("/tmp","opticks",NULL);
    132                }
    133                LOG(verbose) << "expandvar replacing OPTICKS_EVENT_BASE  with " << evalue ;
    134            }







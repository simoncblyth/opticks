ubuntu-integration-test-missing-csgpath-resulting-in-universe-assert
======================================================================


ISSUE reported by Sam on Ubuntu : universe assert
-----------------------------------------------------

* https://groups.io/g/opticks/message/87


Extracts from my mail to him::

  > HepRandomEngine::put called -- no effect! 
      this is harmless Geant4 noise

  > OKG4Test: /home/ubuntu/opticks/cfg4/CTestDetector.cc:243: G4VPhysicalVolume* CTestDetector::makeDetector_NCSG(): Assertion `universe' failed.
     failing to create the universe is a new issue.  


> ... so it would be good to place the log at some url and link to it in your message. 
> I can then diff your log with what I get and see what we learn.
>
>      NCSGList=ERROR GGeoTest=ERROR CTestDetector=ERROR ts box -D
> 
> The envvars are increasing the default output level of those classes to ERROR from
> the usual DEBUG, which will cause them to output in red.


Extracts from Sam's log elided for clarity
-----------------------------------------------

Cause of problem was that the communication from bash/python to C++ was 
assuming a TMP envvar. 

Have fixed this by avoiding that assumption.


::

    === o-main : gdb --args /usr/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero -D 
           --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --test 
           --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=$TMP/tboolean-box_mode=PyCsgInBox_...
                                                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
    ////////////// Culprit found : commandline passing in the csgpath is depending on TMP envvar being defined 

    Starting program: /usr/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero -D --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 
           --test 
           --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-box_outerfirst=1_analytic=1_csgpath=/tboolean-box_mode=PyCsgInBox_...
    
    //////////////  looks like my transition to passing in csgname rather than the path is not complete

    2019-07-04 15:14:29.190 ERROR [12297] [GGeo::init@425]  idpath /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1 cache_exists 1 cache_requested 1 m_loaded_from_cache 1 m_live 0 will_load_libs 1
    2019-07-04 15:14:29.621 INFO  [12297] [NMeta::dump@129] GGeo::loadCacheMeta
    {
        "argline": "/usr/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml --csgskiplv 22 --runfolder geocache-j1808-v5 --runcomment fix-lv10-coincidence-speckle ",
        "location": "Opticks::updateCacheMeta",
        "runcomment": "fix-lv10-coincidence-speckle",
        "rundate": "20190704_150906",
        "runfolder": "geocache-j1808-v5",
        "runlabel": "R0_cvd_",
        "runstamp": 1562252946
    }
    2019-07-04 15:14:29.621 INFO  [12297] [NMeta::dump@129] GGeo::loadCacheMeta.lv2sd
    2019-07-04 15:14:29.637 INFO  [12297] [OpticksHub::loadGeometry@517] --test modifying geometry
    2019-07-04 15:14:29.638 ERROR [12297] [NCSGList::Load@39] missing csgpath /tboolean-box

    ///////////////  not finding the testgeometry at csgpath

    2019-07-04 15:14:29.638 ERROR [12297] [GGeoTest::init@140] [
    2019-07-04 15:14:29.638 ERROR [12297] [GGeoTest::initCreateCSG@230] m_csgpath /tboolean-box



::

    0983 tboolean-box-(){  $FUNCNAME- | python $* ; }
     984 tboolean-box--(){ cat << EOP 
     985 import logging
     986 log = logging.getLogger(__name__)
     987 from opticks.ana.main import opticks_main
     988 from opticks.analytic.polyconfig import PolyConfig
     989 from opticks.analytic.csg import CSG  
     990 
     991 # 0x3f is all 6 
     992 autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
     993 args = opticks_main(csgname="${FUNCNAME/--}", autoemitconfig=autoemitconfig)
     994 
     995 #emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
     996 #emitconfig = "photons:1,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
     997 emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 
     998 
     999 CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, containerautosize=1, emitconfig=emitconfig  )
    1000 
    1001 container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 ) 
    1002 
    1003 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2"  )
    1004 
    1005 CSG.Serialize([container, box], args )
    1006 EOP
    1007 }



CSG.Serialize is culprit for assuming TMP envvar exists
----------------------------------------------------------

analytic/csg.py::

     421     @classmethod
     422     def Serialize(cls, trees, args, outerfirst=1):
     423         """
     424         :param trees: list of CSG instances of solid root nodes
     425         :param args: namespace instance provided by opticks_main directory to save the tree serializations, under an indexed directory 
     426         :param outerfirst: when 1 signifies that the first listed tree contains is the outermost volume 
     427 
     428         1. saves each tree into a separate directories
     429         2. saves FILENAME csg.txt containing boundary strings at top level
     430         3. saves METANAME csgmeta.json containing tree level metadata at top level
     431 
     432         """
     433         #base = args.csgpath 
     434         assert args.csgpath is None, (args.csgpath, args.csgname, "args.csgpath no longer used, replace with csgname=${FUNCNAME/--} " )
     435         base = "$TMP/%s" % args.csgname
     436 
     437         assert type(trees) is list
     438         assert type(base) is str and len(base) > 5, ("invalid base directory %s " % base)
     439         base = os.path.expandvars(base)
     440         log.info("CSG.Serialize : writing %d trees to directory %s " % (len(trees), base))
     441         if not os.path.exists(base):
     442             os.makedirs(base)
     443         pass



Fix to not require TMP envvar 
---------------------------------------------------------------

Could require users to set TMP envvar, but its preferable to be sensitive to TMP 
envvar but not to require it...  which means python and C++ need to have a matched 
default /tmp/USERNAME/opticks

* implemented this in ana/main.py 
* C++ side is in BFile




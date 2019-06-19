:prep-for-shared-opticks-installation
=======================================

Context
----------

* :doc:`metadata-review` 
* :doc:`strace-monitor-file-opens.rst`

Looking into metadata using strace to check opens reveals problems
for shared use of Opticks executables.

example command to strace
-------------------------------

::

    tboolean-;TBOOLEAN_TAG=5 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --strace


is dynamic define still needed ? /home/blyth/local/opticks/gl/dynamic.h
-----------------------------------------------------------------------------

::

    // see boostrap-/BDynamicDefine::write invoked by Scene::write App::prepareScene 
    #define MAXREC 10
    #define MAXTIME 20
    #define PNUMQUAD 4
    #define RNUMQUAD 2
    #define MATERIAL_COLOR_OFFSET 0
    #define FLAG_COLOR_OFFSET 64
    #define PSYCHEDELIC_COLOR_OFFSET 96
    #define SPECTRAL_COLOR_OFFSET 256


relative writes in test running ?
---------------------------------------

::

     tboolean-box/GItemList/GMaterialLib.txt                                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     tboolean-box/GItemList/GSurfaceLib.txt                                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     ## 
     ## oops writes relative to invoking directory even with OPTICKS_EVENT_BASE envvar defind as /tmp  
     ## actually was probably expecting csgpath to be absolute
     ##  
     ## TODO: fix by following the resource approach 

     Culprit::

        115     m_csgpath(m_config->getCSGPath()),
        ... 
        587 void GGeoTest::assignBoundaries()
        588 {
        ...
        611     // see notes/issues/material-names-wrong-python-side.rst
        612     LOG(level) << "Save mlib/slib names "
        613               << " numVolume : " << numVolume
        614               << " csgpath : " << m_csgpath
        615               ;
        616 
        617     if( numVolume > 0 )
        618     {
        619         m_mlib->saveNames(m_csgpath);
        620         m_slib->saveNames(m_csgpath);
        621     }
        622 
        623     LOG(level) << "]" ;
        624 }



::

     831 tboolean-box--(){ cat << EOP 
     832 import logging
     833 log = logging.getLogger(__name__)
     834 from opticks.ana.main import opticks_main
     835 from opticks.analytic.polyconfig import PolyConfig
     836 from opticks.analytic.csg import CSG  
     837 
     838 # 0x3f is all 6 
     839 autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
     840 args = opticks_main(csgpath="${FUNCNAME/--}", autoemitconfig=autoemitconfig)

     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ most tboolean-x--  use $TMP/$FUNCNAME 
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ hmm maybe use that as input to decide where to put the events ? call it CSGFOLD as its the container of the eg tboolean-box dir  
     
     841 
     844 emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 
     845 
     846 CSG.kwa = dict(poly="IM",resolution="20", verbosity="0", ctrl=0, containerscale=3.0, emitconfig=emitconfig  )
     847 
     848 container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container="1" switches on auto-sizing
     849 
     850 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2"  )
     851 
     852 CSG.Serialize([container, box], args )
     853 EOP
     854 }



::

    [blyth@localhost issues]$ tboolean-box-- | python 2>/dev/null | tr "_" "\n"
    autoseqmap=TO:0,SR:1,SA:0
    name=tboolean-box
    outerfirst=1
    analytic=1
    csgpath=tboolean-box
    mode=PyCsgInBox
    autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2
    autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0
    autocontainer=Rock//perfectAbsorbSurface/Vacuum


Running the above serializes the python defined CSG geometry into tboolean-box using relative write from python::

    [blyth@localhost issues]$ l tboolean-box/
    total 8
    -rw-rw-r--. 1 blyth blyth 424 Jun 19 20:04 csgmeta.json
    -rw-rw-r--. 1 blyth blyth  56 Jun 19 20:04 csg.txt
    drwxrwxr-x. 3 blyth blyth 136 Jun 19 20:03 1
    drwxrwxr-x. 3 blyth blyth 136 Jun 19 20:03 0
    [blyth@localhost issues]$ 

So the GGeoTest::assignBoundaries is being consistent with that.

::

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
     433         base = args.csgpath
     434 
     435         assert type(trees) is list
     436         assert type(base) is str and len(base) > 5, ("invalid base directory %s " % base)
     437         base = os.path.expandvars(base)
     438         log.info("CSG.Serialize : writing %d trees to directory %s " % (len(trees), base))
     439         if not os.path.exists(base):
     440             os.makedirs(base)
     441         pass
     442         for it, tree in enumerate(trees):
     443             treedir = cls.treedir(base,it)
     444             tree.save(treedir)
     445         pass
     446 




Approach ?
~~~~~~~~~~~~~~~

1. assert that the csgpath is not absolute and always place it under 




ISSUE : Opticks writes multiple files into install locations
-----------------------------------------------------------------
 
Running from home directory::

     0 Wed Jun 19 16:56:51 CST 2019
     strace.py -f O_CREAT


     /home/blyth/local/opticks/gl/dynamic.h                                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
      ## hmm this will be a problem in shared installation situation

     /tmp/.gl5Q2tTP                                                                   :     O_RDWR|O_CREAT|O_EXCL :  0600 
     ## from OpenGL ?

     /var/tmp/OptixCache/cache.db                                                     :            O_RDWR|O_CREAT :  0666 
     /var/tmp/OptixCache/cache.db                                                     : O_WRONLY|O_CREAT|O_APPEND :  0666 
     /var/tmp/OptixCache/cache.db-journal                                             :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-wal                                                 :            O_RDWR|O_CREAT :  0664 
     /var/tmp/OptixCache/cache.db-shm                                                 :            O_RDWR|O_CREAT :  0664 
     ## need to use OPTIX_CACHE_PATH to control this

     /tmp/blyth/location/seq.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/his.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/mat.npy                                                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/CRandomEngine_jump_photons.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/blyth/location/cg4/primary.npy                                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     ## this is some code still using $TMP /tmp/blyth/location     
     ## maybe get rid of OPTICKS_EVENT_BASE and use $TMP for everything when test running
     ## because it has become misnamed     

     .. event arrays 

     /tmp/tboolean-box/evt/tboolean-box/torch/-5/ht.npy                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/fdom.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/idom.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/parameters.json                      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/parameters.json      :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/Time.ini                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/DeltaTime.ini                        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/VM.ini                               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/DeltaVM.ini                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/Opticks.npy                          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/report.txt                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/OpticksEvent_launch.ini              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/OpticksEvent_prelaunch.ini           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/Time.ini             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/DeltaTime.ini        :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/VM.ini               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/DeltaVM.ini          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/Opticks.npy          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/report.txt           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/OpticksEvent_launch.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/20190619_165615/OpticksEvent_prelaunch.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /tmp/tboolean-box/evt/tboolean-box/torch/-5/History_SequenceSource.json          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/History_SequenceLocal.json           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/Material_SequenceSource.json         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/-5/Material_SequenceLocal.json          :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     persisted indices, maybe this predates better json : TODO revisit index persisting  

        Source looks like counts and Local looks like top of the pops
        blyth@localhost optickscore]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/-5/History_SequenceSource.json
        {
            "4bbcd": "1",
            "86d": "29",
            "8b6ccd": "1",
            "8bd": "6313",
             ...
            "8c6bcd": "1",
            "8cc6ccd": "3",
            "8cc6d": "1",
            "8ccd": "87782",
            "bbbbbb6bcd": "1",
            "bbbbbbb6cd": "9"

        }
        [blyth@localhost optickscore]$ cat /tmp/tboolean-box/evt/tboolean-box/torch/-5/History_SequenceLocal.json
        {
            "4bbcd": "27",
            "4bcd": "19",
            "4cd": "8",
            "86cbcd": "14",



     /tmp/tboolean-box/evt/tboolean-box/torch/5/Time.ini                              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/DeltaTime.ini                         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/VM.ini                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/DeltaVM.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/Opticks.npy                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/report.txt                            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/OpticksEvent_launch.ini               :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/OpticksEvent_prelaunch.ini            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     Probably this profiling is duplicating everything in all ?

     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/Time.ini              :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/DeltaTime.ini         :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/VM.ini                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/DeltaVM.ini           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/Opticks.npy           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/report.txt            :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/OpticksEvent_launch.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/5/20190619_165615/OpticksEvent_prelaunch.ini :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/Time.ini                                :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/DeltaTime.ini                           :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/VM.ini                                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/DeltaVM.ini                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     /tmp/tboolean-box/evt/tboolean-box/torch/Opticks.npy                             :  O_WRONLY|O_CREAT|O_TRUNC :  0666 

     /home/blyth/.opticks/tboolean-box/State/000.ini                                  :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     bookmarks

     /tmp/blyth/location/imgui.ini                                                    :  O_WRONLY|O_CREAT|O_TRUNC :  0666 
     another use of $TMP








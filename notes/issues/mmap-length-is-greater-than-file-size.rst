mmap-length-is-greater-than-file-size
=========================================


Getting this, usually when there are other problems::

    tboolean-boxx --oktest
    ...

    2019-09-23 17:28:26,770] p73031 {__init__            :metadata.py:81} INFO     - path /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/OpticksEvent_launch.ini does not exist 
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/tboolean.py", line 51, in <module>
        ab = AB(ok)
      File "/home/blyth/opticks/ana/ab.py", line 324, in __init__
        self.load()
      File "/home/blyth/opticks/ana/ab.py", line 361, in load
        b = Evt(tag=btag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="B", smry=args.smry)
      File "/home/blyth/opticks/ana/evt.py", line 262, in __init__
        self.init() 
      File "/home/blyth/opticks/ana/evt.py", line 279, in init
        self.init_photons()
      File "/home/blyth/opticks/ana/evt.py", line 499, in init_photons
        ox = self.aload("ox",optional=True) 
      File "/home/blyth/opticks/ana/evt.py", line 379, in aload
        a = A.load_(stem, self.src, self.tag, self.det, optional=optional, dbg=self.dbg, pfx=self.pfx, msli=msli ) 
      File "/home/blyth/opticks/ana/nload.py", line 276, in load_
        arr = np.load(path, mmap_mode="r")  
      File "/home/blyth/anaconda2/lib/python2.7/site-packages/numpy/lib/npyio.py", line 418, in load
        return format.open_memmap(file, mode=mmap_mode)
      File "/home/blyth/anaconda2/lib/python2.7/site-packages/numpy/lib/format.py", line 802, in open_memmap
        mode=mode, offset=offset)
      File "/home/blyth/anaconda2/lib/python2.7/site-packages/numpy/core/memmap.py", line 264, in __new__
        mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
    ValueError: mmap length is greater than file size
    2019-09-23 17:28:26.895 INFO  [72708] [SSys::run@91] tboolean.py --tagoffset 0 --tag 1 --cat tboolean-boxx --pfx tboolean-boxx --src torch --show   rc_raw : 256 rc : 1
    2019-09-23 17:28:26.896 ERROR [72708] [SSys::run@98] FAILED with  cmd tboolean.py --tagoffset 0 --tag 1 --cat tboolean-boxx --pfx tboolean-boxx --src torch --show   RC 1


This is from trying to load non-existing G4 photons::


    ip  tboolean.py --tagoffset 0 --tag 1 --cat tboolean-boxx --pfx tboolean-boxx --src torch --show 

    ipdb> u
    > /home/blyth/opticks/ana/nload.py(276)load_()
        274                 arr = np.load(path)
        275             else:
    --> 276                 arr = np.load(path, mmap_mode="r")
        277                 oshape = arr.shape        #
        278                 arr = arr[msli]

    ipdb> p path
    '/home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/ox.npy'
    ipdb> 


Actually it does exist, but just as a header with no content::

    [blyth@localhost issues]$ xxd -l $(( 16*5 )) -c 16  /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/ox.npy
    0000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    0000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    0000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    0000030: 652c 2027 7368 6170 6527 3a20 2831 3030  e, 'shape': (100
    0000040: 3030 302c 2034 2c20 3429 2c20 7d20 200a  000, 4, 4), }  .

    [blyth@localhost issues]$ l /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/ox.npy
    -rw-rw-r--. 1 blyth blyth 80 Sep 23 17:28 /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/ox.npy



Problem is trying to use AB when there is no B.  Would be best for OpticksEvent to skip the -1
dir rather than filling it with partials. Then would be easy to detect::


    blyth@localhost issues]$ cd /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1/
    [blyth@localhost -1]$ l
    total 84
    drwxrwxr-x. 2 blyth blyth  239 Sep 23 17:28 20190923_172824
    -rw-rw-r--. 1 blyth blyth 2256 Sep 23 17:28 DeltaVM.ini
    -rw-rw-r--. 1 blyth blyth   80 Sep 23 17:28 OpticksProfileAccLabels.npy
    -rw-rw-r--. 1 blyth blyth   80 Sep 23 17:28 OpticksProfileAcc.npy
    -rw-rw-r--. 1 blyth blyth 4944 Sep 23 17:28 OpticksProfileLabels.npy
    -rw-rw-r--. 1 blyth blyth 1296 Sep 23 17:28 OpticksProfile.npy
    -rw-rw-r--. 1 blyth blyth 9217 Sep 23 17:28 report.txt
    -rw-rw-r--. 1 blyth blyth 2206 Sep 23 17:28 DeltaTime.ini
    -rw-rw-r--. 1 blyth blyth 2977 Sep 23 17:28 VM.ini
    -rw-rw-r--. 1 blyth blyth 2567 Sep 23 17:28 Time.ini
    -rw-rw-r--. 1 blyth blyth  128 Sep 23 17:28 fdom.npy
    -rw-rw-r--. 1 blyth blyth  176 Sep 23 17:28 gs.npy
    -rw-rw-r--. 1 blyth blyth   96 Sep 23 17:28 idom.npy
    -rw-rw-r--. 1 blyth blyth   80 Sep 23 17:28 ox.npy
    -rw-rw-r--. 1 blyth blyth 2045 Sep 23 17:28 parameters.json
    -rw-rw-r--. 1 blyth blyth   80 Sep 23 17:28 ph.npy
    -rw-rw-r--. 1 blyth blyth   96 Sep 23 17:28 rx.npy
    -rw-rw-r--. 1 blyth blyth   27 Sep 23 17:28 gs.json
    -rw-rw-r--. 1 blyth blyth   80 Sep 23 17:28 ht.npy
    drwxrwxr-x. 2 blyth blyth  239 Sep 23 17:26 20190923_172609
    [blyth@localhost -1]$ 



Hmm not enough to use OKTest need also --nog4propagate.

* TODO: maybe can auto set this when using OKTest 


::

    086 void OpticksRun::createEvent(unsigned tagoffset)
     87 {
     88     bool nog4propagate = m_ok->isNoG4Propagate() ;   // --nog4propagate
     89 
     90     m_ok->setTagOffset(tagoffset);
     91     // tagoffset is recorded with Opticks::setTagOffset within the makeEvent, but need it here before that 
     92 
     93     OK_PROFILE("_OpticksRun::createEvent");
     94 
     95 
     96     m_evt = m_ok->makeEvent(true, tagoffset) ;
     97     std::string tstamp = m_evt->getTimeStamp();
     98 
     99     if(nog4propagate)
    100     {
    101         m_g4evt = NULL ;
    102     }
    103     else
    104     {
    105         m_g4evt = m_ok->makeEvent(false, tagoffset) ;
    106         m_g4evt->setSibling(m_evt);
    107         m_g4evt->setTimeStamp( tstamp.c_str() );   // align timestamps
    108         m_evt->setSibling(m_g4evt);
    109     }
    110 


::

    ts boxx --oktest 

::

    [blyth@localhost evtbase]$ l  /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1
    ls: cannot access /home/blyth/local/opticks/evtbase/tboolean-boxx/evt/tboolean-boxx/torch/-1: No such file or directory


Added the --nog4propagate as a result of --oktest to tboolean-lv::

    [blyth@localhost evtbase]$ t tboolean-lv
    tboolean-lv is a function
    tboolean-lv () 
    { 
        local msg="=== $FUNCNAME :";
        local funcname=$(tboolean-funcname);
        local testname=$(tboolean-testname);
        local RC;
        echo $msg $testname;
        local cmdline="$*";
        if [ "${cmdline/--ip}" != "${cmdline}" ]; then
            TESTNAME=$testname tboolean-ipy- $*;
        else
            if [ "${cmdline/--py}" != "${cmdline}" ]; then
                TESTNAME=$testname tboolean-py- $*;
            else
                if [ "${cmdline/--chk}" != "${cmdline}" ]; then
                    ${funcname}-;
                else
                    if [ "${cmdline/--oktest}" != "${cmdline}" ]; then
                        $funcname $* --nog4propagate;
                    else
                        if [ "${cmdline/--noalign}" != "${cmdline}" ]; then
                            $funcname --okg4test $*;
                        else
                            $funcname --okg4test --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $*;
                            RC=$?;
                        fi;
                    fi;
                fi;
            fi;
        fi;
        echo $msg $funcname RC $RC;
        return $RC
    }



Somehow AB not then invoked,


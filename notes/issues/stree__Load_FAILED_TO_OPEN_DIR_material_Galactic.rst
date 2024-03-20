stree__Load_FAILED_TO_OPEN_DIR_material_Galactic
===================================================


Hmm Galactic again. I recall this previously happening 
due to a material with no properties resulting in not creating 
a fold. 

Kludge fix used previously::

    N[blyth@localhost ~]$ mkdir /data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic
    N[blyth@localhost ~]$ cd /data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic
    N[blyth@localhost Galactic]$ touch NPFold_index.txt
    N[blyth@localhost Galactic]$ touch NPFold_names.txt


Added NPFold debug and existance check, see::

   ~/np/tests/NPFold_Load_test.sh

   export NPFold__load_DUMP=1
   export NPFold__load_index_DUMP=1
   export NPFold__load_dir_DUMP=1


Changed NPFold::load to just skip empty folders.


::


    N[blyth@localhost ~]$ ~/o/sysrap/tests/SScene_test.sh build_dbg 


    NPFold__load_dir_DUMP=1 ~/o/sysrap/tests/SScene_test.sh build_dbg 



    U::DirList FAILED TO OPEN DIR /data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic

    Program received signal SIGINT, Interrupt.
    0x00007ffff752d387 in raise () from /lib64/libc.so.6
    (gdb) r
    The program being debugged has been started already.
    Start it from the beginning? (y or n) n
    Program not restarted.
    (gdb) bt
    #0  0x00007ffff752d387 in raise () from /lib64/libc.so.6
    #1  0x00000000004097b0 in U::DirList (names=..., path=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic", ext=0x0, 
        exclude=false) at ../NPU.hh:1404
    #2  0x0000000000418172 in NPFold::load_dir (this=0x265b230, 
        _base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic") at ../NPFold.h:1834
    #3  0x00000000004184e0 in NPFold::load (this=0x265b230, _base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic")
        at ../NPFold.h:1916
    #4  0x0000000000415d3b in NPFold::Load_ (base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic") at ../NPFold.h:450
    #5  0x0000000000415eca in NPFold::Load (base_=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material", rel_=0x4c44f0 "Galactic")
        at ../NPFold.h:493
    #6  0x0000000000418101 in NPFold::load_subfold (this=0x4c3550, _base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material", 
        relp=0x4c44f0 "Galactic") at ../NPFold.h:1750
    #7  0x0000000000418374 in NPFold::load_index (this=0x4c3550, _base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material")
        at ../NPFold.h:1877
    #8  0x00000000004184cb in NPFold::load (this=0x4c3550, _base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material")
        at ../NPFold.h:1916
    #9  0x0000000000415d3b in NPFold::Load_ (base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material") at ../NPFold.h:450
    #10 0x0000000000415eca in NPFold::Load (base_=0x4c0760 "/data/blyth/opticks/U4TreeCreateTest/stree", rel_=0x4c3450 "material")
        at ../NPFold.h:493
    #11 0x0000000000418101 in NPFold::load_subfold (this=0x4c0510, _base=0x4c0760 "/data/blyth/opticks/U4TreeCreateTest/stree", 
        relp=0x4c3450 "material") at ../NPFold.h:1750
    #12 0x0000000000418374 in NPFold::load_index (this=0x4c0510, _base=0x4c0760 "/data/blyth/opticks/U4TreeCreateTest/stree")
        at ../NPFold.h:1877
    #13 0x00000000004184cb in NPFold::load (this=0x4c0510, _base=0x4c0760 "/data/blyth/opticks/U4TreeCreateTest/stree") at ../NPFold.h:1916
    #14 0x0000000000415d3b in NPFold::Load_ (base=0x4c0760 "/data/blyth/opticks/U4TreeCreateTest/stree") at ../NPFold.h:450
    #15 0x0000000000415e90 in NPFold::Load (base_=0x4c0490 "/data/blyth/opticks/U4TreeCreateTest/stree") at ../NPFold.h:488
    #16 0x000000000042f386 in stree::load_ (this=0x4bfc20, dir=0x4c0490 "/data/blyth/opticks/U4TreeCreateTest/stree") at ../stree.h:2265
    #17 0x000000000042f31b in stree::load (this=0x4bfc20, base=0x4799f6 "$STREE_FOLD", reldir=0x4799f0 "stree") at ../stree.h:2258
    #18 0x000000000042f2b3 in stree::Load (base=0x4799f6 "$STREE_FOLD", reldir=0x4799f0 "stree") at ../stree.h:2252
    #19 0x0000000000405bf6 in main () at SScene_test.cc:15
    (gdb) f 17
    #17 0x000000000042f31b in stree::load (this=0x4bfc20, base=0x4799f6 "$STREE_FOLD", reldir=0x4799f0 "stree") at ../stree.h:2258
    2258	    int rc = load_(dir); 
    (gdb) f 19
    #19 0x0000000000405bf6 in main () at SScene_test.cc:15
    15	    stree* st = stree::Load("$STREE_FOLD"); 
    (gdb) 



    (gdb) f 7
    #7  0x0000000000418374 in NPFold::load_index (this=0x4c3550, _base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material")
        at ../NPFold.h:1877
    1877	            load_subfold(_base, key);  // instanciates NPFold and add_subfold
    (gdb) p key
    $1 = 0x4c44f0 "Galactic"
    (gdb) f 6
    #6  0x0000000000418101 in NPFold::load_subfold (this=0x4c3550, _base=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material", 
        relp=0x4c44f0 "Galactic") at ../NPFold.h:1750
    1750	    add_subfold(relp,  NPFold::Load(_base, relp) ) ; 
    (gdb) f 5
    #5  0x0000000000415eca in NPFold::Load (base_=0x1680230 "/data/blyth/opticks/U4TreeCreateTest/stree/material", rel_=0x4c44f0 "Galactic")
        at ../NPFold.h:493
    493	    return Load_(base); 
    (gdb) p base
    $2 = 0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic"
    (gdb) f 4
    #4  0x0000000000415d3b in NPFold::Load_ (base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic") at ../NPFold.h:450
    450	    nf->load(base); 
    (gdb) f 3
    #3  0x00000000004184e0 in NPFold::load (this=0x265b230, _base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic")
        at ../NPFold.h:1916
    1916	    int rc = has_index ? load_index(_base) : load_dir(_base) ; 
    (gdb) p has_index
    $3 = false
    (gdb) f 2
    #2  0x0000000000418172 in NPFold::load_dir (this=0x265b230, 
        _base=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic") at ../NPFold.h:1834
    1834	    U::DirList(names, base) ; 
    (gdb) f 1
    #1  0x00000000004097b0 in U::DirList (names=..., path=0x265b060 "/data/blyth/opticks/U4TreeCreateTest/stree/material/Galactic", ext=0x0, 
        exclude=false) at ../NPU.hh:1404
    1404	    if(!dir && RAISE) std::raise(SIGINT) ; 
    (gdb) 




FIXED U4Tree_stree_snd_scsg_FAIL_consistent_parent
=====================================================

issue fixed first with 1 and then with 2 

1. scsg::init IMAX reserving to prevent reallocations
2. avoid using snd* pointers after snd::Add in snd::Boolean 
   to set the parent : AS REALLOCATION CAN HAPPEN AT ANY PUSH_BACK 


See: 

* sysrap/tests/stree_load_test.sh 


::

    epsilon:tests blyth$ ./stree_load_test.sh run > stree_load_test.log
    epsilon:tests blyth$ grep FAIL stree_load_test.log
    snd::desc  FAIL consistent_parent_index  ch 0 count 0 child.parent -1 nd.index 2 nd.lvid 0 child.index 0 child.lvid 0
    snd::desc  FAIL consistent_parent_index  ch 1 count 1 child.parent -1 nd.index 2 nd.lvid 0 child.index 1 child.lvid 0
    snd::desc  FAIL consistent_parent_index  ch 14 count 0 child.parent -1 nd.index 16 nd.lvid 6 child.index 14 child.lvid 6
    snd::desc  FAIL consistent_parent_index  ch 15 count 1 child.parent -1 nd.index 16 nd.lvid 6 child.index 15 child.lvid 6
    snd::desc  FAIL consistent_parent_index  ch 62 count 0 child.parent -1 nd.index 64 nd.lvid 24 child.index 62 child.lvid 24
    snd::desc  FAIL consistent_parent_index  ch 63 count 1 child.parent -1 nd.index 64 nd.lvid 24 child.index 63 child.lvid 24
    snd::desc  FAIL consistent_parent_index  ch 62 count 0 child.parent -1 nd.index 64 nd.lvid 24 child.index 62 child.lvid 24
    snd::desc  FAIL consistent_parent_index  ch 63 count 1 child.parent -1 nd.index 64 nd.lvid 24 child.index 63 child.lvid 24
    snd::desc  FAIL consistent_parent_index  ch 254 count 0 child.parent -1 nd.index 256 nd.lvid 62 child.index 254 child.lvid 62
    snd::desc  FAIL consistent_parent_index  ch 255 count 1 child.parent -1 nd.index 256 nd.lvid 62 child.index 255 child.lvid 62
    epsilon:tests blyth$ 


Notice how the issue shows up at close to power of two node indices : that made be look for realloc bugs. 



Which solids are effected is kinda random::

    In [2]: t.soname[0]
    Out[2]: 'sTopRock_domeAir0x59e3e90'

    In [3]: t.soname[6]
    Out[3]: 'Upper_Tyvek_tube0x73ff920'

    In [4]: t.soname[24]
    Out[4]: 'GLw1.up04_up05_FlangeI_Web_FlangeII0x5a02eb0'

    In [5]: t.soname[62]
    Out[5]: 'GLb3.bt11_FlangeI_Web_FlangeII0x5a6c1d0'


::

    In [17]: label,n[n[:,lv]==0]
    Out[17]: 
    ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
     array([[  0,   1,   0, *-1,*  0,  -1,   1,   0, 105,   0,   0,  -1],     105:CSG_CYLINDER
            [  1,   1,   1, *-1,*  0,  -1,  -1,   0, 110,   1,   1,   0],     110:CSG_BOX3
            [  2,   0,  -1,  -1,   2,   0,  -1,   0,   3,  -1,  -1,  -1]], dtype=int32))

    [ U4Tree::Create 
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y lvid  -1 type   5 root    0 U4Solid::Tag(type) Tub
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y lvid  -1 type   4 root    1 U4Solid::Tag(type) Box
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y lvid  -1 type  14 root    1 U4Solid::Tag(type) Dis
    snd::Boolean change l 0 ln.parent -1 to 2
    snd::Boolean change r 1 rn.parent -1 to 2
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y lvid   0 type  13 root    2 U4Solid::Tag(type) Sub
    U4Solid::init SUCCEEDED desc: U4Solid::desc solid Y lvid  -1 type   5 root    3 U4Solid::Tag(type) Tub

    ## parent "2" failing to get set  
    In [24]: n[n[:,pt]==2]
    Out[24]: array([], shape=(0, 12), dtype=int32)



Issue manifests as unset parent for a sprinkling (<10) of Boolean child nodes out of >600::

        In [18]: label,n[n[:,lv]==6]
        Out[18]: 
        ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
         array([[ 14,   1,   0, *-1,*  0,  -1,  15,   6, 105,  10,  10,  -1],
                [ 15,   1,   1, *-1,*  0,  -1,  -1,   6, 105,  11,  11,  -1],
                [ 16,   0,  -1,  -1,   2,  14,  -1,   6,   3,  -1,  -1,  -1]], dtype=int32))

        In [19]: label,n[n[:,lv]==24]
        Out[19]: 
        ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
         array([[ 62,   2,   0, *-1,*  0,  -1,  63,  24, 110,  43,  43,  -1],
                [ 63,   2,   1, *-1,*  0,  -1,  -1,  24, 110,  44,  44,  17],
                [ 64,   1,   0,  66,   2,  62,  65,  24,   1,  -1,  -1,  -1],
                [ 65,   1,   1,  66,   0,  -1,  -1,  24, 110,  45,  45,  18],
                [ 66,   0,  -1,  -1,   2,  64,  -1,  24,   1,  -1,  -1,  -1]], dtype=int32))

        In [16]: label,n[n[:,lv]==62]
        Out[16]: 
        ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
         array([[252,   2,   0, 254,   0,  -1, 253,  62, 110, 157, 157,  -1],
                [253,   2,   1, 254,   0,  -1,  -1,  62, 110, 158, 158,  93],
                [254,   1,   0, *-1,*  2, 252, 255,  62,   1,  -1,  -1,  -1],
                [255,   1,   1, *-1,*  0,  -1,  -1,  62, 110, 159, 159,  94],
                [256,   0,  -1,  -1,   2, 254,  -1,  62,   1,  -1,  -1,  -1]], dtype=int32))


        ## dp > 0 but pt still -1 

        In [42]: label,n[np.logical_and(n[:,dp]>0,n[:,pt]==-1)]
        Out[42]: 
        ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
         array([[  0,   1,   0,  -1,   0,  -1,   1,   0, 105,   0,   0,  -1],
                [  1,   1,   1,  -1,   0,  -1,  -1,   0, 110,   1,   1,   0],
                [ 14,   1,   0,  -1,   0,  -1,  15,   6, 105,  10,  10,  -1],
                [ 15,   1,   1,  -1,   0,  -1,  -1,   6, 105,  11,  11,  -1],
                [ 62,   2,   0,  -1,   0,  -1,  63,  24, 110,  43,  43,  -1],
                [ 63,   2,   1,  -1,   0,  -1,  -1,  24, 110,  44,  44,  17],
                [254,   1,   0,  -1,   2, 252, 255,  62,   1,  -1,  -1,  -1],
                [255,   1,   1,  -1,   0,  -1,  -1,  62, 110, 159, 159,  94]], dtype=int32))




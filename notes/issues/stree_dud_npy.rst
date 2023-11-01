stree_dud_npy : SOMEHOW MISREPORTED ISSUE : AS THE EMPTIES NOT CURRENTLY CAUSING ISSUE
==========================================================================================

stree is persisting zero length NP.hh arrays for all those.
Which python numpy takes exception to.

* not now : it must have been something else ? maybe NumPy version dependency ? 


AHHA: those are placeholder arrays with the real info the sidecar _names.txt

HMM : to avoid numpy errors should perhaps use an array with length 1
rather than 0 ?



Python NumPy writes header only ?
-----------------------------------

::

    epsilon:nn blyth$ i
    Using matplotlib backend: MacOSX

    In [1]: a = np.array([])
    In [2]: a
    Out[2]: array([], dtype=float64)
    In [3]: np.save("empty.npy",a)
    In [4]:
    epsilon:nn blyth$ l
    total 8
    8 -rw-r--r--   1 blyth  wheel   128 Nov  1 19:55 empty.npy
    0 drwxr-xr-x   3 blyth  wheel    96 Nov  1 19:55 .
    0 drwxrwxrwt  38 root   wheel  1216 Nov  1 19:54 ..
    epsilon:nn blyth$ xxd empty.npy
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6638 272c 2027 666f 7274  r': '<f8', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    epsilon:nn blyth$


::

    epsilon:nn blyth$ i
    Using matplotlib backend: MacOSX
    In [1]: b = np.load("empty.npy")
    In [2]: b
    Out[2]: array([], dtype=float64)

    epsilon:stree blyth$ pwd
    /Users/blyth/.opticks/GEOM/V1J011/CSGFoundry/SSim/stree
    epsilon:stree blyth$ l | grep 128
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 sensor_name.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 subs.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 digs.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 implicit.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 soname.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 suname.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 mtline.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 mtname.npy
        8 -rw-rw-r--   1 blyth  staff       128 Oct 30 12:33 mtname_no_rindex.npy
    epsilon:stree blyth$

    epsilon:stree blyth$ l *.txt
        8 -rw-rw-r--  1 blyth  staff       300 Oct 30 12:33 NPFold_index.txt
        8 -rw-rw-r--  1 blyth  staff        98 Oct 30 12:33 sensor_name_names.txt
    24888 -rw-rw-r--  1 blyth  staff  12741696 Oct 30 12:33 subs_names.txt
    24888 -rw-rw-r--  1 blyth  staff  12741696 Oct 30 12:33 digs_names.txt
       16 -rw-rw-r--  1 blyth  staff      5436 Oct 30 12:33 implicit_names.txt
        8 -rw-rw-r--  1 blyth  staff      3809 Oct 30 12:33 soname_names.txt
       16 -rw-rw-r--  1 blyth  staff      6836 Oct 30 12:33 suname_names.txt
        8 -rw-rw-r--  1 blyth  staff       173 Oct 30 12:33 mtname_names.txt
        8 -rw-rw-r--  1 blyth  staff       117 Oct 30 12:33 mtname_no_rindex_names.txt
    epsilon:stree blyth$



::

    epsilon:stree blyth$ find . -size 128c -name '*.npy' -exec xxd {} \;
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c20  e, 'shape': (0,
    00000040: 3129 2c20 7d20 2020 2020 2020 2020 2020  1), }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
    00000010: 7227 3a20 273c 6934 272c 2027 666f 7274  r': '<i4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2830 2c29  e, 'shape': (0,)
    00000040: 2c20 7d20 2020 2020 2020 2020 2020 2020  , }
    00000050: 2020 2020 2020 2020 2020 2020 2020 2020
    00000060: 2020 2020 2020 2020 2020 2020 2020 2020
    00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
    epsilon:stree blyth$




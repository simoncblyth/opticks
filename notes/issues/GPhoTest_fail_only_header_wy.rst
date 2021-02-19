GPhoTest_fail_only_header_wy
===============================


Since disabling "--way" by default G4OKTest is producing .npy with a header but no content.

* TODO: skip saving the wy.npy when way is not enabled 



::

    epsilon:opticks blyth$ l /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy
    -rw-r--r--  1 blyth  wheel  80 Feb 18 12:09 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy

    epsilon:opticks blyth$ xxd /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2835 3030  e, 'shape': (500
    00000040: 302c 2032 2c20 3429 2c20 7d20 2020 200a  0, 2, 4), }    .
    epsilon:opticks blyth$ 


::

    [blyth@localhost ggeo]$ ls -l $TMP/G4OKTest/evt/g4live/natural/1/ox.npy $TMP/G4OKTest/evt/g4live/natural/1/wy.npy
    -rw-rw-r--. 1 blyth blyth 320080 Feb 20 05:18 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox.npy
    -rw-rw-r--. 1 blyth blyth     80 Feb 20 05:18 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy
    [blyth@localhost ggeo]$ date
    Sat Feb 20 05:39:01 CST 2021
    [blyth@localhost ggeo]$ 

    epsilon:issues blyth$ ls -l $TMP/G4OKTest/evt/g4live/natural/1/ox.npy $TMP/G4OKTest/evt/g4live/natural/1/wy.npy
    -rw-r--r--  1 blyth  wheel  320080 Feb 19 21:40 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/ox.npy
    -rw-r--r--  1 blyth  wheel      80 Feb 19 21:40 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy






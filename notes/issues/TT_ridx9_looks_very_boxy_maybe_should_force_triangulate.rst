TT_ridx9_looks_very_boxy_maybe_should_force_triangulate
===========================================================

The compound solid (ridx 9) is composed entirely of boxes, so a triangulated
representation would be perfectly accurate. BUT the force triangulated
mode is not implemented for instanced geometry yet. 

Dumping ridx 9, revealing that its all boxes::

    P[blyth@localhost sysrap]$ TEST=get_repeat_node RIDX=9 RORD=0 ~/o/sysrap/tests/stree_load_test.sh run
    stree::init 
    stree::load_ /home/blyth/.opticks/GEOM/J_2024aug27/CSGFoundry/SSim/stree
    stree::desc_repeat_nodes q_repeat_index 9 q_repeat_ordinal 0 num_node 130
    snode ix:     15 dh: 7 sx:    0 pt:     14 nc:    1 fc:     16 ns:    145 lv: 11 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:11 sn:-1 sPanel
    snode ix:     16 dh: 8 sx:    0 pt:     15 nc:   64 fc:     17 ns:     -1 lv: 10 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:12 sn:-1 sPanelTape
    snode ix:     17 dh: 9 sx:    0 pt:     16 nc:    1 fc:     18 ns:     19 lv:  9 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     18 dh:10 sx:    0 pt:     17 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     19 dh: 9 sx:    1 pt:     16 nc:    1 fc:     20 ns:     21 lv:  9 cp:      1 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     20 dh:10 sx:    0 pt:     19 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     21 dh: 9 sx:    2 pt:     16 nc:    1 fc:     22 ns:     23 lv:  9 cp:      2 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     22 dh:10 sx:    0 pt:     21 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     23 dh: 9 sx:    3 pt:     16 nc:    1 fc:     24 ns:     25 lv:  9 cp:      3 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     24 dh:10 sx:    0 pt:     23 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     25 dh: 9 sx:    4 pt:     16 nc:    1 fc:     26 ns:     27 lv:  9 cp:      4 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     26 dh:10 sx:    0 pt:     25 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     27 dh: 9 sx:    5 pt:     16 nc:    1 fc:     28 ns:     29 lv:  9 cp:      5 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     28 dh:10 sx:    0 pt:     27 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     29 dh: 9 sx:    6 pt:     16 nc:    1 fc:     30 ns:     31 lv:  9 cp:      6 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     30 dh:10 sx:    0 pt:     29 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     31 dh: 9 sx:    7 pt:     16 nc:    1 fc:     32 ns:     33 lv:  9 cp:      7 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     32 dh:10 sx:    0 pt:     31 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     33 dh: 9 sx:    8 pt:     16 nc:    1 fc:     34 ns:     35 lv:  9 cp:      8 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     34 dh:10 sx:    0 pt:     33 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     35 dh: 9 sx:    9 pt:     16 nc:    1 fc:     36 ns:     37 lv:  9 cp:      9 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     36 dh:10 sx:    0 pt:     35 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     37 dh: 9 sx:   10 pt:     16 nc:    1 fc:     38 ns:     39 lv:  9 cp:     10 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     38 dh:10 sx:    0 pt:     37 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     39 dh: 9 sx:   11 pt:     16 nc:    1 fc:     40 ns:     41 lv:  9 cp:     11 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     40 dh:10 sx:    0 pt:     39 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     41 dh: 9 sx:   12 pt:     16 nc:    1 fc:     42 ns:     43 lv:  9 cp:     12 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     42 dh:10 sx:    0 pt:     41 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     43 dh: 9 sx:   13 pt:     16 nc:    1 fc:     44 ns:     45 lv:  9 cp:     13 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     44 dh:10 sx:    0 pt:     43 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     45 dh: 9 sx:   14 pt:     16 nc:    1 fc:     46 ns:     47 lv:  9 cp:     14 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     46 dh:10 sx:    0 pt:     45 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     47 dh: 9 sx:   15 pt:     16 nc:    1 fc:     48 ns:     49 lv:  9 cp:     15 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     48 dh:10 sx:    0 pt:     47 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     49 dh: 9 sx:   16 pt:     16 nc:    1 fc:     50 ns:     51 lv:  9 cp:     16 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     50 dh:10 sx:    0 pt:     49 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     51 dh: 9 sx:   17 pt:     16 nc:    1 fc:     52 ns:     53 lv:  9 cp:     17 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     52 dh:10 sx:    0 pt:     51 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     53 dh: 9 sx:   18 pt:     16 nc:    1 fc:     54 ns:     55 lv:  9 cp:     18 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     54 dh:10 sx:    0 pt:     53 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     55 dh: 9 sx:   19 pt:     16 nc:    1 fc:     56 ns:     57 lv:  9 cp:     19 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     56 dh:10 sx:    0 pt:     55 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     57 dh: 9 sx:   20 pt:     16 nc:    1 fc:     58 ns:     59 lv:  9 cp:     20 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     58 dh:10 sx:    0 pt:     57 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     59 dh: 9 sx:   21 pt:     16 nc:    1 fc:     60 ns:     61 lv:  9 cp:     21 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     60 dh:10 sx:    0 pt:     59 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     61 dh: 9 sx:   22 pt:     16 nc:    1 fc:     62 ns:     63 lv:  9 cp:     22 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     62 dh:10 sx:    0 pt:     61 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     63 dh: 9 sx:   23 pt:     16 nc:    1 fc:     64 ns:     65 lv:  9 cp:     23 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     64 dh:10 sx:    0 pt:     63 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     65 dh: 9 sx:   24 pt:     16 nc:    1 fc:     66 ns:     67 lv:  9 cp:     24 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     66 dh:10 sx:    0 pt:     65 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     67 dh: 9 sx:   25 pt:     16 nc:    1 fc:     68 ns:     69 lv:  9 cp:     25 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     68 dh:10 sx:    0 pt:     67 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     69 dh: 9 sx:   26 pt:     16 nc:    1 fc:     70 ns:     71 lv:  9 cp:     26 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     70 dh:10 sx:    0 pt:     69 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     71 dh: 9 sx:   27 pt:     16 nc:    1 fc:     72 ns:     73 lv:  9 cp:     27 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     72 dh:10 sx:    0 pt:     71 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     73 dh: 9 sx:   28 pt:     16 nc:    1 fc:     74 ns:     75 lv:  9 cp:     28 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     74 dh:10 sx:    0 pt:     73 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     75 dh: 9 sx:   29 pt:     16 nc:    1 fc:     76 ns:     77 lv:  9 cp:     29 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     76 dh:10 sx:    0 pt:     75 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     77 dh: 9 sx:   30 pt:     16 nc:    1 fc:     78 ns:     79 lv:  9 cp:     30 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     78 dh:10 sx:    0 pt:     77 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     79 dh: 9 sx:   31 pt:     16 nc:    1 fc:     80 ns:     81 lv:  9 cp:     31 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     80 dh:10 sx:    0 pt:     79 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     81 dh: 9 sx:   32 pt:     16 nc:    1 fc:     82 ns:     83 lv:  9 cp:     32 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     82 dh:10 sx:    0 pt:     81 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     83 dh: 9 sx:   33 pt:     16 nc:    1 fc:     84 ns:     85 lv:  9 cp:     33 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     84 dh:10 sx:    0 pt:     83 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     85 dh: 9 sx:   34 pt:     16 nc:    1 fc:     86 ns:     87 lv:  9 cp:     34 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     86 dh:10 sx:    0 pt:     85 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     87 dh: 9 sx:   35 pt:     16 nc:    1 fc:     88 ns:     89 lv:  9 cp:     35 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     88 dh:10 sx:    0 pt:     87 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     89 dh: 9 sx:   36 pt:     16 nc:    1 fc:     90 ns:     91 lv:  9 cp:     36 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     90 dh:10 sx:    0 pt:     89 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     91 dh: 9 sx:   37 pt:     16 nc:    1 fc:     92 ns:     93 lv:  9 cp:     37 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     92 dh:10 sx:    0 pt:     91 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     93 dh: 9 sx:   38 pt:     16 nc:    1 fc:     94 ns:     95 lv:  9 cp:     38 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     94 dh:10 sx:    0 pt:     93 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     95 dh: 9 sx:   39 pt:     16 nc:    1 fc:     96 ns:     97 lv:  9 cp:     39 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     96 dh:10 sx:    0 pt:     95 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     97 dh: 9 sx:   40 pt:     16 nc:    1 fc:     98 ns:     99 lv:  9 cp:     40 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:     98 dh:10 sx:    0 pt:     97 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:     99 dh: 9 sx:   41 pt:     16 nc:    1 fc:    100 ns:    101 lv:  9 cp:     41 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    100 dh:10 sx:    0 pt:     99 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    101 dh: 9 sx:   42 pt:     16 nc:    1 fc:    102 ns:    103 lv:  9 cp:     42 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    102 dh:10 sx:    0 pt:    101 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    103 dh: 9 sx:   43 pt:     16 nc:    1 fc:    104 ns:    105 lv:  9 cp:     43 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    104 dh:10 sx:    0 pt:    103 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    105 dh: 9 sx:   44 pt:     16 nc:    1 fc:    106 ns:    107 lv:  9 cp:     44 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    106 dh:10 sx:    0 pt:    105 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    107 dh: 9 sx:   45 pt:     16 nc:    1 fc:    108 ns:    109 lv:  9 cp:     45 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    108 dh:10 sx:    0 pt:    107 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    109 dh: 9 sx:   46 pt:     16 nc:    1 fc:    110 ns:    111 lv:  9 cp:     46 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    110 dh:10 sx:    0 pt:    109 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    111 dh: 9 sx:   47 pt:     16 nc:    1 fc:    112 ns:    113 lv:  9 cp:     47 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    112 dh:10 sx:    0 pt:    111 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    113 dh: 9 sx:   48 pt:     16 nc:    1 fc:    114 ns:    115 lv:  9 cp:     48 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    114 dh:10 sx:    0 pt:    113 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    115 dh: 9 sx:   49 pt:     16 nc:    1 fc:    116 ns:    117 lv:  9 cp:     49 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    116 dh:10 sx:    0 pt:    115 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    117 dh: 9 sx:   50 pt:     16 nc:    1 fc:    118 ns:    119 lv:  9 cp:     50 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    118 dh:10 sx:    0 pt:    117 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    119 dh: 9 sx:   51 pt:     16 nc:    1 fc:    120 ns:    121 lv:  9 cp:     51 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    120 dh:10 sx:    0 pt:    119 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    121 dh: 9 sx:   52 pt:     16 nc:    1 fc:    122 ns:    123 lv:  9 cp:     52 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    122 dh:10 sx:    0 pt:    121 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    123 dh: 9 sx:   53 pt:     16 nc:    1 fc:    124 ns:    125 lv:  9 cp:     53 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    124 dh:10 sx:    0 pt:    123 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    125 dh: 9 sx:   54 pt:     16 nc:    1 fc:    126 ns:    127 lv:  9 cp:     54 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    126 dh:10 sx:    0 pt:    125 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    127 dh: 9 sx:   55 pt:     16 nc:    1 fc:    128 ns:    129 lv:  9 cp:     55 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    128 dh:10 sx:    0 pt:    127 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    129 dh: 9 sx:   56 pt:     16 nc:    1 fc:    130 ns:    131 lv:  9 cp:     56 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    130 dh:10 sx:    0 pt:    129 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    131 dh: 9 sx:   57 pt:     16 nc:    1 fc:    132 ns:    133 lv:  9 cp:     57 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    132 dh:10 sx:    0 pt:    131 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    133 dh: 9 sx:   58 pt:     16 nc:    1 fc:    134 ns:    135 lv:  9 cp:     58 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    134 dh:10 sx:    0 pt:    133 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    135 dh: 9 sx:   59 pt:     16 nc:    1 fc:    136 ns:    137 lv:  9 cp:     59 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    136 dh:10 sx:    0 pt:    135 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    137 dh: 9 sx:   60 pt:     16 nc:    1 fc:    138 ns:    139 lv:  9 cp:     60 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    138 dh:10 sx:    0 pt:    137 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    139 dh: 9 sx:   61 pt:     16 nc:    1 fc:    140 ns:    141 lv:  9 cp:     61 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    140 dh:10 sx:    0 pt:    139 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    141 dh: 9 sx:   62 pt:     16 nc:    1 fc:    142 ns:    143 lv:  9 cp:     62 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    142 dh:10 sx:    0 pt:    141 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0
    snode ix:    143 dh: 9 sx:   63 pt:     16 nc:    1 fc:    144 ns:     -1 lv:  9 cp:     63 se:     -1 se:     -1 ri: 9 ro:    0 bd:13 sn:-1 sBar_1
    snode ix:    144 dh:10 sx:    0 pt:    143 nc:    0 fc:     -1 ns:     -1 lv:  8 cp:      0 se:     -1 se:     -1 ri: 9 ro:    0 bd:14 sn:-1 sBar_0

    ulvid {8,9,10,11,}
    stree::desc_solid lvid 8 lvn sBar_0 root Y sn::rbrief
      0 : sn::brief tc  110 cm  0 lv   8 xf N pa Y bb Y pt N nc  0 dp  0 tg bo

    stree::desc_solid lvid 9 lvn sBar_1 root Y sn::rbrief
      0 : sn::brief tc  110 cm  0 lv   9 xf N pa Y bb Y pt N nc  0 dp  0 tg bo

    stree::desc_solid lvid 10 lvn sPanelTape root Y sn::rbrief
      0 : sn::brief tc  110 cm  0 lv  10 xf N pa Y bb Y pt N nc  0 dp  0 tg bo

    stree::desc_solid lvid 11 lvn sPanel root Y sn::rbrief
      0 : sn::brief tc  110 cm  0 lv  11 xf N pa Y bb Y pt N nc  0 dp  0 tg bo



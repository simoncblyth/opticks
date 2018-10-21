flight_path_view
==================


Designing Paths
-----------------

Aiming to design flight paths externally with NumPy rather than thru bookmark interpolation, 
so need a scratch global geometry in matplotlib.

Added volume index to GParts idxBuffer, which is populated by X4PhysicalVolume::convertNode
Am most interested in this for large global volumes, for the "scratch" big picture geometry::

   
    epsilon:0 blyth$ np.py 
    /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1/GParts/0
            ./GParts.txt : 219 
         ./idxBuffer.npy :             (201, 4) : 3e6f8c55de891e502afb5ac6c94ff0d0 
        ./partBuffer.npy :          (219, 4, 4) : f70b99bb26556b7941530525e9090b6b 
        ./tranBuffer.npy :       (206, 3, 4, 4) : 4bb4281e353d702c2937fe51b08d0ac3 
        ./primBuffer.npy :             (201, 4) : cd3222aea6b1292bd3382340b61e1d62 
    epsilon:0 blyth$ np.py idxBuffer.npy 
    (201, 4)


   ('i32\n', 
array([[     0,     39,     39,      0],
       [     1,     12,     12,      0],
       [     2,     11,     11,      0],
       [     3,      3,      3,      0],
       [     4,      0,      0,      0],
       [     5,      1,      1,      1],
       [     6,      2,      2,      1],
       [     7,     10,     10,      1],
       [ 61545,      9,      9,      0],
       [ 62067,      8,      8,      0],
       [ 62067,      8,      8,      0],
       [ 61545,      9,      9,      0],
       [ 62067,      8,      8,      0],
       [ 62067,      8,      8,      0],
       [ 61545,      9,      9,      0],
       [ 62067,      8,      8,      0],
       [ 62067,      8,      8,      0],
       [ 61545,      9,      9,      0],
       [ 62067,      8,      8,      0],
       [ 62067,      8,      8,      0],
       [ 61545,      9,      9,      0],
       ...
       ... hmm repeated volume index ??? : but am not very interested in those 

       [ 62067,      8,      8,      0],
       [ 62067,      8,      8,      0],
       [ 62588,     38,     38,      0],
       [ 62589,     37,     37,      0],
       [ 62590,     36,     36,      0],
       [ 62591,     35,     35,      1],
       [ 62592,     34,     34,      1],
       [ 62593,     14,     14,      1],
       [ 62594,     13,     13,      1],
       [352849,     31,     31,      0],
       [352850,     28,     28,      1],
       [352851,     29,     29,      0],
       [352852,     30,     30,      1],
       [352853,     33,     33,      0],
       [352854,     32,     32,      0]], dtype=int32))
 


::

    epsilon:ana blyth$ ./mm0prim.py 
    INFO:__main__:/usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1
    WARNING:opticks.ana.mesh:using IDPATH from environment
    INFO:opticks.ana.mesh:Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1 
    INFO:opticks.ana.mesh:loading map /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1/MeshIndex/GItemIndexSource.json kv pairs 40 
    primIdx   0 idx                  [ 0 39 39  0] lvIdx  39 lvName                sWorld0x4bc2350 partOffset   0 numParts   1 tranOffset   0 numTran   1 planOffset   0  
    primIdx   1 idx                  [ 1 12 12  0] lvIdx  12 lvName              sTopRock0x4bccfc0 partOffset   1 numParts   1 tranOffset   1 numTran   1 planOffset   0  
    primIdx   2 idx                  [ 2 11 11  0] lvIdx  11 lvName              sExpHall0x4bcd390 partOffset   2 numParts   1 tranOffset   2 numTran   1 planOffset   0  
    primIdx   3 idx                      [3 3 3 0] lvIdx   3 lvName         Upper_Chimney0x5b2e8e0 partOffset   3 numParts   1 tranOffset   3 numTran   1 planOffset   0  
    primIdx   4 idx                      [4 0 0 0] lvIdx   0 lvName         Upper_LS_tube0x5b2e9f0 partOffset   4 numParts   1 tranOffset   4 numTran   1 planOffset   0  
    primIdx 188 idx      [62588    38    38     0] lvIdx  38 lvName           sBottomRock0x4bcd770 partOffset 194 numParts   1 tranOffset 189 numTran   1 planOffset   0  
    primIdx 189 idx      [62589    37    37     0] lvIdx  37 lvName           sPoolLining0x4bd1eb0 partOffset 195 numParts   1 tranOffset 190 numTran   1 planOffset   0  
    primIdx 190 idx      [62590    36    36     0] lvIdx  36 lvName       sOuterWaterPool0x4bd2960 partOffset 196 numParts   1 tranOffset 191 numTran   1 planOffset   0  
    primIdx 195 idx  [352849     31     31      0] lvIdx  31 lvName            sWaterTube0x5b30eb0 partOffset 209 numParts   1 tranOffset 200 numTran   1 planOffset   0  
    primIdx 197 idx  [352851     29     29      0] lvIdx  29 lvName            sChimneyLS0x5b312e0 partOffset 213 numParts   1 tranOffset 202 numTran   1 planOffset   0  
    primIdx 199 idx  [352853     33     33      0] lvIdx  33 lvName             sSurftube0x5b3ab80 partOffset 217 numParts   1 tranOffset 204 numTran   1 planOffset   0  
    primIdx 200 idx  [352854     32     32      0] lvIdx  32 lvName          svacSurftube0x5b3bf50 partOffset 218 numParts   1 tranOffset 205 numTran   1 planOffset   0  
    epsilon:ana blyth$ 



Establish correspondence to volume indices::

    In [5]: run mm0prim.py
    [2018-10-21 14:46:46,338] p28014 {/Users/blyth/opticks/ana/mm0prim.py:27} INFO - /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1
    [2018-10-21 14:46:46,341] p28014 {/Users/blyth/opticks/ana/mesh.py:33} WARNING - using IDPATH from environment
    [2018-10-21 14:46:46,341] p28014 {/Users/blyth/opticks/ana/mesh.py:38} INFO - Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1 
    [2018-10-21 14:46:46,341] p28014 {/Users/blyth/opticks/ana/mesh.py:42} INFO - loading map /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1/MeshIndex/GItemIndexSource.json kv pairs 40 
    primIdx   0 idx                  [ 0 39 39  0] lvIdx  39 lvName                sWorld0x4bc2350 partOffset   0 numParts   1 tranOffset   0 numTran   1 planOffset   0  
    lWorld0x4bc2710_PV
    lWorld0x4bc2710
    primIdx   1 idx                  [ 1 12 12  0] lvIdx  12 lvName              sTopRock0x4bccfc0 partOffset   1 numParts   1 tranOffset   1 numTran   1 planOffset   0  
    pTopRock0x4bcd120
    lTopRock0x4bcd050
    primIdx   2 idx                  [ 2 11 11  0] lvIdx  11 lvName              sExpHall0x4bcd390 partOffset   2 numParts   1 tranOffset   2 numTran   1 planOffset   0  
    pExpHall0x4bcd520
    lExpHall0x4bcd420
    primIdx   3 idx                      [3 3 3 0] lvIdx   3 lvName         Upper_Chimney0x5b2e8e0 partOffset   3 numParts   1 tranOffset   3 numTran   1 planOffset   0  
    lUpperChimney_phys0x5b308a0
    lUpperChimney0x5b2ed40
    primIdx   4 idx                      [4 0 0 0] lvIdx   0 lvName         Upper_LS_tube0x5b2e9f0 partOffset   4 numParts   1 tranOffset   4 numTran   1 planOffset   0  
    pUpperChimneyLS0x5b2f160
    lUpperChimneyLS0x5b2ee40
    primIdx 188 idx      [62588    38    38     0] lvIdx  38 lvName           sBottomRock0x4bcd770 partOffset 194 numParts   1 tranOffset 189 numTran   1 planOffset   0  
    pBtmRock0x4bd2650
    lBtmRock0x4bd1df0
    primIdx 189 idx      [62589    37    37     0] lvIdx  37 lvName           sPoolLining0x4bd1eb0 partOffset 195 numParts   1 tranOffset 190 numTran   1 planOffset   0  
    pPoolLining0x4bd25b0
    lPoolLining0x4bd24f0
    primIdx 190 idx      [62590    36    36     0] lvIdx  36 lvName       sOuterWaterPool0x4bd2960 partOffset 196 numParts   1 tranOffset 191 numTran   1 planOffset   0  
    pOuterWaterPool0x4bd2b70
    lOuterWaterPool0x4bd2a70
    primIdx 195 idx  [352849     31     31      0] lvIdx  31 lvName            sWaterTube0x5b30eb0 partOffset 209 numParts   1 tranOffset 200 numTran   1 planOffset   0  
    lLowerChimney_phys0x5b32c20
    lLowerChimney0x5b30fc0
    primIdx 197 idx  [352851     29     29      0] lvIdx  29 lvName            sChimneyLS0x5b312e0 partOffset 213 numParts   1 tranOffset 202 numTran   1 planOffset   0  
    pLowerChimneyLS0x5b317e0
    lLowerChimneyLS0x5b313f0
    primIdx 199 idx  [352853     33     33      0] lvIdx  33 lvName             sSurftube0x5b3ab80 partOffset 217 numParts   1 tranOffset 204 numTran   1 planOffset   0  
    lSurftube_phys0x5b3c810
    lSurftube0x5b3ac50
    primIdx 200 idx  [352854     32     32      0] lvIdx  32 lvName          svacSurftube0x5b3bf50 partOffset 218 numParts   1 tranOffset 205 numTran   1 planOffset   0  
    pvacSurftube0x5b3c120
    lvacSurftube0x5b3c020




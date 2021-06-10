analysis_shakedown
=====================

::

   PFX=tds3gun ab.sh 1 



would be good to see the double step data in local frame : need nodeIndex for that to access the transform
-------------------------------------------------------------------------------------------------------------

tds3ip.sh::

    In [3]: a.ox[:,3].view(np.uint32)                                                                                                                                                                  
    Out[3]: 
    A([[4293394432,     126601,          0,       6152],
       [4293342850,     126774,          1,      22592],
       [4293342937,     127122,          2,       6272],
       [4293394432,     126949,          3,       6152],
       [4293332465,      85234,          4,      38976],
       [4293332509,      85410,          5,      22592],
       [4293201524,      85758,          6,       6272],
       [   1179648,      67843,          7,       4104]], dtype=uint32)

From okc/OpticksPhotonFlags::

    x: bit packed : Boundary and SensorIndex 
    y: NodeIndex of last intersect 
    z: photon index
    w: flag mask (derivable from the seqhis)  


oxrap/cu/photon.h::

    152 __device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset)
    153 {
    154     pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );
    155     pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
    156     pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
    157     pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w);
    158 }
    159 

oxrap/cu/generate.cu::

    842 
    843     // setting p.flags for things like boundary, history flags  
    844     FLAGS(p, s, prd);
    845 
    846     p.flags.u.z = photon_id ;  // formerly behind IDENTITY_DEBUG macro, but has become indispensable
    847 


    211 /**2
    212 FLAGS Macro 
    213 ------------
    214 
    215 Sets the photon flags p.flags using values from state s and per-ray-data prd
    216 
    217 p.flags.u.x 
    218    packed signed int boundary and unsigned sensorIndex which are 
    219    assumed to fit in 16 bits into 32 bits, see SPack::unsigned_as_int 
    220 
    221 p.flags.u.y
    222    now getting s.identity.x (nodeIndex) thanks to the packing 
    223 
    224 s.identity.x
    225     node index 
    226 
    227 s.identity.w 
    228     sensor index arriving from GVolume::getIdentity.w
    229 
    230 ::
    231 
    232     256 glm::uvec4 GVolume::getIdentity() const
    233     257 {
    234     258     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    235     259     return id ;
    236     260 }
    237 
    238 NumPy array access::
    239 
    240     boundary    = (( flags[:,0].view(np.uint32) & 0xffff0000 ) >> 16 ).view(np.int16)[1::2] 
    241     sensorIndex = (( flags[:,0].view(np.uint32) & 0x0000ffff ) >>  0 ).view(np.int16)[0::2] 
    242 
    243 
    244 Formerly::
    245 
    246     p.flags.i.x = prd.boundary ;  \
    247     p.flags.u.y = s.identity.w ;  \
    248     p.flags.u.w |= s.flag ; \
    249 
    250 2**/
    251 
    252 #define FLAGS(p, s, prd) \
    253 { \
    254     p.flags.u.x = ( ((prd.boundary & 0xffff) << 16) | (s.identity.w & 0xffff) )  ;  \
    255     p.flags.u.y = s.identity.x ;  \
    256     p.flags.u.w |= s.flag ; \
    257 } \
    258 
    259 






But other than w the G4 pflags are very different. Whats going on ?::

    In [4]: b.ox[:,3].view(np.uint32)                                                                                                                                                                  
    Out[4]: 
    A([[       5,        0, 67305984,     6272],
       [       5,        0, 67305985,     6304],
       [       5,        0, 67305986,     6208],
       [       1,        0, 67305987,     4104],
       [       7,        0, 67305988,     6240],
       [       5,        0, 67305989,     6272],
       [       3,        0, 67305990,     6152],
       [       5,        0, 67305991,     7328]], dtype=uint32)



    377 void CWriter::writePhoton_(const G4StepPoint* point, unsigned record_id  )
    378 {
    ...
    396     m_photons_buffer->setUInt(record_id, 3, 0, 0, m_photon._slot_constrained );
    397     m_photons_buffer->setUInt(record_id, 3, 0, 1, 0u );
    398     m_photons_buffer->setUInt(record_id, 3, 0, 2, m_photon._c4.u );
    399     m_photons_buffer->setUInt(record_id, 3, 0, 3, m_photon._mskhis );
    400 }


Whats the G4 way to get the nodeIndex ? Would be useful to label all the volumes.
Some JUNO volumes use the CopyNo.

g4-cls G4PVPlacement::

    125     inline G4int GetCopyNo() const  { return fcopyNo; }
    126 
    127     void  SetCopyNo(G4int CopyNo);
    128       // Gets and sets the copy number of the volume.


Vague recollection that GGeo geometry conversion holds onto void* of source volumes::

    152 /**
    153 GVolume::getOriginNode
    154 ------------------------
    155 
    156 *OriginNode* set in ctor is used to record the G4VPhysicalVolume from whence the GVolume 
    157 was converted, see X4PhysicalVolume::convertNode
    158 
    159 **/
    160 
    161 void* GVolume::getOriginNode() const
    162 {
    163     return m_origin_node ;
    164 }
    165 

    epsilon:ggeo blyth$ opticks-f getOriginNode 
    ./extg4/X4PhysicalVolume.cc:        const void* const sensorOrigin = sensor->getOriginNode(); 
    ./extg4/X4PhysicalVolume.cc:        const void* const outerOrigin = outer->getOriginNode(); 
    ./ggeo/GNodeLib.cc:            const void* const sensorOrigin = sensor->getOriginNode() ;
    ./ggeo/GNodeLib.cc:            const void* const outerOrigin = outer->getOriginNode() ;  
    ./ggeo/GNodeLib.cc:            origin = outer->getOriginNode() ;  
    ./ggeo/GNodeLib.cc:            origin = sensor->getOriginNode() ;  
    ./ggeo/GVolume.cc:GVolume::getOriginNode
    ./ggeo/GVolume.cc:void* GVolume::getOriginNode() const 
    ./ggeo/GVolume.hh:      void*        getOriginNode() const ;
    epsilon:opticks blyth$ 


Can create a mapping to get from an originNode to the GVolume. GNodeLib is the place to do this.
This kinda think might be expensive (std::map of 300k items) 
but its only for debugging/validation so no worries. 

But having local positions in general frames is not so useful, the local position
in the instance frame is what is useful.


From jsd, can get the transform::

     370     std::string volname = track->GetVolume()->GetName(); // physical volume
     371     // == position
     372     const G4AffineTransform& trans = track->GetTouchable()->GetHistory()->GetTopTransform();
     373     const G4ThreeVector& global_pos = postStepPoint->GetPosition();
     374     G4ThreeVector local_pos = trans.TransformPoint(global_pos);
     375 








overall shapes : G4 ht and gs arrays not populated
-----------------------------------------------------

* are missing the postPropagate call 

so.npy is missing for a:OK and empty for b:G4::

    epsilon:ana blyth$ l /tmp/blyth/opticks/tds3gun/evt/g4live/natural/{1,2,-1,-2}/so.npy
    ls: /tmp/blyth/opticks/tds3gun/evt/g4live/natural/1/so.npy: No such file or directory
    ls: /tmp/blyth/opticks/tds3gun/evt/g4live/natural/2/so.npy: No such file or directory
    8 -rw-rw-r--  1 blyth  wheel  80 Jun  9 15:10 /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-2/so.npy
    8 -rw-rw-r--  1 blyth  wheel  80 Jun  9 15:10 /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-1/so.npy
    epsilon:ana blyth$ 


::

    In [1]: a                                                                                                                                                                                                
    Out[1]: 
    Evt(  1,"natural","g4live",pfx="tds3gun", seqs="[]", msli="0:100k:" ) 20210609-1526 
    /tmp/blyth/opticks/tds3gun/evt/g4live/natural/1
     file_photons 11278   load_slice 0:100k:   loaded_photons 11278 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:3000000  
         gs :            - :       66,6,4 : (gensteps) 
         ox :    11278,4,4 :    11278,4,4 : (photons) final photon step   
         wl :            - :        11278 : (photons) wavelength 
       post :            - :      11278,4 : (photons) final photon step: position, time 
       dirw :            - :      11278,4 : (photons) final photon step: direction, weight  
       polw :            - :      11278,4 : (photons) final photon step: polarization, wavelength  
     pflags :            - :        11278 : (photons) final photon step: flags  
         c4 :            - :        11278 : (photons) final photon step: dtype split uint8 view of ox flags 
         ht :            - :     3421,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :         3421 : (hits) wavelength 
      hpost :            - :       3421,4 : (hits) final photon step: position, time 
      hdirw :            - :       3421,4 : (hits) final photon step: direction, weight  
      hpolw :            - :       3421,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :         3421 : (hits) final photon step: flags  
        hc4 :            - :         3421 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx : 11278,10,2,4 : 11278,10,2,4 : (records) photon step records 
         ph :    11278,1,2 :    11278,1,2 : (records) photon history flag/material sequence 
         so :            - :              : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 

    In [2]: b                                                                                                                                                                                                
    Out[2]: 
    Evt( -1,"natural","g4live",pfx="tds3gun", seqs="[]", msli="0:100k:" ) 20210609-1526 
    /tmp/blyth/opticks/tds3gun/evt/g4live/natural/-1
     file_photons 11278   load_slice 0:100k:   loaded_photons 11278 
       fdom :            - :        3,1,4 : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
       idom :            - :        1,1,4 : (metadata) maxbounce:9 maxrec:10 maxrng:3000000  
         gs :            - :              : (gensteps) 
         ox :    11278,4,4 :    11278,4,4 : (photons) final photon step   
         wl :            - :        11278 : (photons) wavelength 
       post :            - :      11278,4 : (photons) final photon step: position, time 
       dirw :            - :      11278,4 : (photons) final photon step: direction, weight  
       polw :            - :      11278,4 : (photons) final photon step: polarization, wavelength  
     pflags :            - :        11278 : (photons) final photon step: flags  
         c4 :            - :        11278 : (photons) final photon step: dtype split uint8 view of ox flags 
         ht :            - :        0,4,4 : (hits) surface detect SD final photon steps 
        hwl :            - :            0 : (hits) wavelength 
      hpost :            - :          0,4 : (hits) final photon step: position, time 
      hdirw :            - :          0,4 : (hits) final photon step: direction, weight  
      hpolw :            - :          0,4 : (hits) final photon step: polarization, wavelength  
     hflags :            - :            0 : (hits) final photon step: flags  
        hc4 :            - :            0 : (hits) final photon step: dtype split uint8 view of ox flags 
         rx : 11278,10,2,4 : 11278,10,2,4 : (records) photon step records 
         ph :    11278,1,2 :    11278,1,2 : (records) photon history flag/material sequence 
         so :        0,4,4 :        0,4,4 : (source) input CPU side emitconfig photons, or initial cerenkov/scintillation 

    In [3]:                                         




G4 Extra BT lines 
---------------------


* thought this could be the virtuals which are skipped in the OK conversion, 
  but seems they are gone from current geometry ? so maybe a difference 
  of handling very close surfaces


Extra BT lines for G4 makes difficult to use the comparison machinery::

    In [17]: a.seqhis_ana.table                                                                                                                                                                              
    Out[17]: 
    all_seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                              11278         1.00 
    0000               42        0.147        1653        [2 ] SI AB
    0001            7ccc2        0.115        1292        [5 ] SI BT BT BT SD
    0002            8ccc2        0.052         590        [5 ] SI BT BT BT SA
    0003           7ccc62        0.052         581        [6 ] SI SC BT BT BT SD
    0004              452        0.037         422        [3 ] SI RE AB
    0005              462        0.035         392        [3 ] SI SC AB
    0006           7ccc52        0.034         380        [6 ] SI RE BT BT BT SD
    0007             4cc2        0.025         278        [4 ] SI BT BT AB
    0008           8ccc62        0.022         251        [6 ] SI SC BT BT BT SA
    0009          7ccc662        0.019         219        [7 ] SI SC SC BT BT BT SD
    0010            4cc62        0.017         197        [5 ] SI SC BT BT AB
    0011          7ccc652        0.014         157        [7 ] SI RE SC BT BT BT SD
    0012           8ccc52        0.014         154        [6 ] SI RE BT BT BT SA
    0013               41        0.013         142        [2 ] CK AB
    0014             4662        0.012         137        [4 ] SI SC SC AB
    0015             4552        0.011         124        [4 ] SI RE RE AB
    0016             4652        0.011         121        [4 ] SI RE SC AB
    0017            4cc52        0.010         117        [5 ] SI RE BT BT AB
    0018          7ccc552        0.009         102        [7 ] SI RE RE BT BT BT SD
    0019           4cc662        0.007          82        [6 ] SI SC SC BT BT AB
    .                              11278         1.00 

    In [18]: b.seqhis_ana.table                                                                                                                                                                              
    Out[18]: 
    all_seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                              11278         1.00 
    0000               42        0.148        1665        [2 ] SI AB
    0001           7cccc2        0.118        1336        [6 ] SI BT BT BT BT SD
    0002          7cccc62        0.053         599        [7 ] SI SC BT BT BT BT SD
    0003           8cccc2        0.052         583        [6 ] SI BT BT BT BT SA
    0004              452        0.047         534        [3 ] SI RE AB
    0005             8cc2        0.041         464        [4 ] SI BT BT SA
    0006          7cccc52        0.038         432        [7 ] SI RE BT BT BT BT SD
    0007              462        0.033         367        [3 ] SI SC AB
    0008          8cccc62        0.022         249        [7 ] SI SC BT BT BT BT SA
    0009         7cccc662        0.020         230        [8 ] SI SC SC BT BT BT BT SD
    0010            8cc62        0.016         186        [5 ] SI SC BT BT SA
    0011         7cccc652        0.015         172        [8 ] SI RE SC BT BT BT BT SD
    0012          8cccc52        0.015         168        [7 ] SI RE BT BT BT BT SA
    0013               41        0.013         144        [2 ] CK AB
    0014            8ccc2        0.013         143        [5 ] SI BT BT BT SA
    0015             4552        0.013         142        [4 ] SI RE RE AB
    0016            8cc52        0.012         138        [5 ] SI RE BT BT SA
    0017         7cccc552        0.012         138        [8 ] SI RE RE BT BT BT BT SD
    0018             4cc2        0.011         127        [4 ] SI BT BT AB
    0019             4662        0.011         121        [4 ] SI SC SC AB
    .                              11278         1.00 

    In [19]:                                                                



    In [19]: a.seqhis_ana.table.compare(b.seqhis_ana.table)[:10]                                                                                                                                             
    [{compare             :seq.py    :628} INFO     - cfordering_key for noshortname?
    Out[19]: 
    noshortname?
    .                  cfo:self  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278      9085.35/74 = 122.78  (pval:0.000 prob:1.000)  
    0000               42      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] SI AB
    0001            7ccc2      1292         0   1292          1292.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] SI BT BT BT SD
    0002            8ccc2       590       143    447           272.59        4.126 +- 0.170        0.242 +- 0.020  [5 ] SI BT BT BT SA
    0003           7ccc62       581         0    581           581.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] SI SC BT BT BT SD
    0004              452       422       534   -112            13.12        0.790 +- 0.038        1.265 +- 0.055  [3 ] SI RE AB
    0005              462       392       367     25             0.82        1.068 +- 0.054        0.936 +- 0.049  [3 ] SI SC AB
    0006           7ccc52       380         0    380           380.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] SI RE BT BT BT SD
    0007             4cc2       278       127    151            56.30        2.189 +- 0.131        0.457 +- 0.041  [4 ] SI BT BT AB
    0008           8ccc62       251        43    208           147.16        5.837 +- 0.368        0.171 +- 0.026  [6 ] SI SC BT BT BT SA
    0009          7ccc662       219         0    219           219.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] SI SC SC BT BT BT SD
    .                              11278     11278      9085.35/74 = 122.78  (pval:0.000 prob:1.000)  

    In [20]:                    





applying selection required py3 map fix
------------------------------------------

::


    In [3]: a.sel = "SI BT BT BT SD"                                                                                                                                                                         

    In [4]: a.seqhis_ana.table                                                                                                                                                                               
    Out[4]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                               1292         1.00 
    0000            7ccc2        1.000        1292        [5 ] SI BT BT BT SD
    .                               1292         1.00 

    In [5]: a.rpostr()                                                                                                                                                                                       
    Out[5]: 
    A([[  438.6035, 17699.3604, 17820.1052, ...,     0.    ,     0.    ,     0.    ],
       [  438.6035, 17700.5492, 17819.4569, ...,     0.    ,     0.    ,     0.    ],
       [  438.6035, 17699.4801, 17819.8775, ...,     0.    ,     0.    ,     0.    ],
       ...,
       [  501.8379, 17699.5081, 17820.4845, ...,     0.    ,     0.    ,     0.    ],
       [  501.8379, 17700.0216, 17819.3663, ...,     0.    ,     0.    ,     0.    ],
       [  501.8379, 17699.3465, 17820.8191, ...,     0.    ,     0.    ,     0.    ]])

    In [6]: a.rpostr().shape                                                                                                                                                                                 
    Out[6]: (1292, 10)



wildcard selection, removed ox.missing check, handled 0 length so
---------------------------------------------------------------------

* again the BT difference prevents proper comparison 

::

    In [3]: a.sel = "CK .."                                                                                                                                                                                  

    In [4]: a.seqhis_ana.table[:20]                                                                                                                                                                          
    Out[4]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                263         1.00 
    0000               41        0.540         142        [2 ] CK AB
    0001              451        0.065          17        [3 ] CK RE AB
    0002           7ccc51        0.049          13        [6 ] CK RE BT BT BT SD
    0003             4551        0.023           6        [4 ] CK RE RE AB
    0004           8ccc51        0.019           5        [6 ] CK RE BT BT BT SA
    0005         4ccccc51        0.015           4        [8 ] CK RE BT BT BT BT BT AB
    0006          7ccc651        0.015           4        [7 ] CK RE SC BT BT BT SD
    0007            4cc51        0.011           3        [5 ] CK RE BT BT AB
    0008            7ccc1        0.011           3        [5 ] CK BT BT BT SD
    0009            8ccc1        0.011           3        [5 ] CK BT BT BT SA
    0010          7ccc551        0.011           3        [7 ] CK RE RE BT BT BT SD
    0011             4651        0.011           3        [4 ] CK RE SC AB
    0012           4cc651        0.008           2        [6 ] CK RE SC BT BT AB
    0013       9999cccc51        0.008           2        [10] CK RE BT BT BT BT DR DR DR DR
    0014            46651        0.008           2        [5 ] CK RE SC SC AB
    0015          4cccc51        0.008           2        [7 ] CK RE BT BT BT BT AB
    0016        4c9cccc51        0.008           2        [9 ] CK RE BT BT BT BT DR BT AB
    0017       c999cccc51        0.008           2        [10] CK RE BT BT BT BT DR DR DR BT
    0018          4666551        0.004           1        [7 ] CK RE RE SC SC SC AB
    0019          7c6cc51        0.004           1        [7 ] CK RE BT BT SC BT SD
    .                                263         1.00 


    In [1]: b.sel = "CK .."                                                                                                                                                                                  

    In [3]: b.seqhis_ana.table[:20]                                                                                                                                                                          
    Out[3]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                263         1.00 
    0000               41        0.548         144        [2 ] CK AB
    0001              451        0.049          13        [3 ] CK RE AB
    0002          7cccc51        0.038          10        [7 ] CK RE BT BT BT BT SD
    0003        7cccc6651        0.023           6        [9 ] CK RE SC SC BT BT BT BT SD
    0004          8cccc51        0.023           6        [7 ] CK RE BT BT BT BT SA
    0005           8cc551        0.019           5        [6 ] CK RE RE BT BT SA
    0006         7cccc651        0.019           5        [8 ] CK RE SC BT BT BT BT SD
    0007         7cccc551        0.019           5        [8 ] CK RE RE BT BT BT BT SD
    0008             4551        0.015           4        [4 ] CK RE RE AB
    0009           8cc651        0.015           4        [6 ] CK RE SC BT BT SA
    0010            8cc51        0.015           4        [5 ] CK RE BT BT SA
    0011             8cc1        0.011           3        [4 ] CK BT BT SA
    0012            46651        0.011           3        [5 ] CK RE SC SC AB
    0013         8cccc651        0.011           3        [8 ] CK RE SC BT BT BT BT SA
    0014           455551        0.008           2        [6 ] CK RE RE RE RE AB
    0015           8cccc1        0.008           2        [6 ] CK BT BT BT BT SA
    0016            8ccc1        0.008           2        [5 ] CK BT BT BT SA
    0017           7cccc1        0.008           2        [6 ] CK BT BT BT BT SD
    0018        7cccc6551        0.008           2        [9 ] CK RE RE SC BT BT BT BT SD
    0019             4cc1        0.008           2        [4 ] CK BT BT AB
    .                                263         1.00 



Best way to investigate the BT difference is with tds3ip input photons
-------------------------------------------------------------------------

::

    epsilon:offline blyth$ PFX=tds3ip ab.sh 1

    als[:10]
    TO BT BT AB
    TO BT BT BT SD
    *TO BT BT BT SA*
    TO BT BT AB
    TO BT BT BT SD
    TO BT BT BT SD
    TO BT BT BT SA
    TO AB

    bls[:10]
    TO BT BT BT BT SA
    TO SC BT BT BT SA
    *TO BT BT BT BT SD*
    TO AB
    TO SC SC BT BT BT BT SD
    TO BT BT BT BT SA
    TO BT BT AB
    TO SC BT BT BR SA



    In [7]: b.rpost_(slice(None))[2]                                                                                                                                                                         
    Out[7]: 
    A([[     0.    ,      0.    ,      0.    ,      0.293 ],  TO
       [-10219.4281,  10219.4281, -10219.4281,     90.9696],  BT
       [-10289.0103,  10289.0103, -10289.0103,     91.5922],  BT
       [-11127.6589,  11127.6589, -11127.6589,     98.2574],  BT
       [-11127.6589,  11127.6589, -11127.6589,     98.2574],  BT  <--- duplicated point from G4
       [-11129.49  ,  11129.49  , -11129.49  ,     98.294 ],  SA 
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ]])

    In [8]: a.rpost_(slice(None))[2]                                                                                                                                                                         
    Out[8]: 
    A([[     0.    ,      0.    ,      0.    ,      0.293 ], TO
       [-10219.4281,  10219.4281, -10219.4281,     90.8963], BT
       [-10289.0103,  10289.0103, -10289.0103,     91.5189], BT
       [-11127.6589,  11127.6589, -11127.6589,     98.1841], BT   <--- only appears once with OK : maybe the miniscule thickness PMT surf ? and float/double diff  
       [-11129.49  ,  11129.49  , -11129.49  ,     98.2208], SA
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ],
       [     0.    ,      0.    ,      0.    ,      0.    ]])



* these are CubeCorners input photons hence x=y=z except for sign
* recall the values here are domain compressed, so in full precision they might just be very close. 
* a 2d ray trace render of geometry would help for this. 
* could suppress such close points in CRecorder ?


* TODO: debug output for photon index 2 
* debug output is ok for initial check but arrays of data is much more lastingly useful, 
  for this problem the most useful thing would be an double precision version of the 
  m_records_buffer -> m_double_buffer 

  * added dx.npy "deluxe double precision" buffer to OpticksEvent 




dx.npy
--------


::

    In [17]: a[2]                                                                                                                                                                                            
    Out[17]: 
    array([[[    -0.577,      0.577,     -0.577,      0.3  ],
            [     0.707,      0.   ,     -0.707,    440.   ]],

           [[-10219.1  ,  10219.1  , -10219.1  ,     90.955],
            [     0.707,      0.   ,     -0.707,    440.   ]],

           [[-10288.382,  10288.382, -10288.382,     91.575],
            [     0.707,      0.   ,     -0.707,    440.   ]],

           [[-11126.93 ,  11126.93 , -11126.93 ,     98.271],
            [     0.822,      0.424,     -0.382,    440.   ]],

           [[-11126.93 ,  11126.93 , -11126.93 ,     98.271],
            [     0.822,      0.424,     -0.382,    440.   ]],

           [[-11129.938,  11130.117, -11129.867,     98.297],
            [     0.822,      0.424,     -0.382,    440.   ]],

           [[     0.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      0.   ,      0.   ]],

           [[     0.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      0.   ,      0.   ]],

           [[     0.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      0.   ,      0.   ]],

           [[     0.   ,      0.   ,      0.   ,      0.   ],
            [     0.   ,      0.   ,      0.   ,      0.   ]]])


Positions of those points 3 and 4 really are exactly the same to double precision, but the time is slightly different::

    In [27]: for i in range(4):print("3:%20.10f 4:%20.10f  4-3:%20.10f" % (a[2,3,0,i], a[2,4,0,i],a[2,4,0,i]-a[2,3,0,i]   ))                                                                                 
    3:   -11126.9296875000 4:   -11126.9296875000  4-3:        0.0000000000
    3:    11126.9296875000 4:    11126.9296875000  4-3:        0.0000000000
    3:   -11126.9296875000 4:   -11126.9296875000  4-3:        0.0000000000
    3:       98.2705612183 4:       98.2705688477  4-3:        0.0000076294




To see whats going on here, can shoot parallel input photons in 1 mm increments around the prime direction.
Then can plot the points. 




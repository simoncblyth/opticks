Natural Gensteps Shakedown
============================

FIXED : by getting G4StepNPY.cc to handle mixed gensteps
----------------------------------------------------------




Issue : no visible prop, all photons missing geometry
----------------------------------------------------------

::

   op --natural 

   op --natural --compute --save
   op --natural --load
 

Appears to run in interop and compute modes, but no visible propagation
and indices shows everything in single categories.
  

::

    simon:ana blyth$ ipython -i $(which tevt.py) --  --tag 1 --det dayabay --src natural
    /Users/blyth/opticks/ana/tevt.py --tag 1 --det dayabay --src natural
    [2016-08-23 18:36:49,558] p80226 {/Users/blyth/opticks/ana/evt.py:99} INFO - loaded metadata from /tmp/blyth/opticks/evt/dayabay/natural/1 :                     /tmp/blyth/opticks/evt/dayabay/natural/1 9fcd324f5d8ebb0888b35fab2df35150 65f79001783636de1f7487072d4824e8 1142140     0.2943 COMPUTE_MODE  
    [2016-08-23 18:36:49,777] p80226 {/Users/blyth/opticks/ana/evt.py:194} WARNING - init_records dayabay/natural/  1 :  finds too few (ph)seqhis uniques : 1 : EMPTY HISTORY
    [2016-08-23 18:36:49,777] p80226 {/Users/blyth/opticks/ana/evt.py:196} WARNING - init_records dayabay/natural/  1 :  finds too few (ph)seqmat uniques : 1 : EMPTY HISTORY
    [2016-08-23 18:36:51,041] p80226 {/Users/blyth/opticks/ana/tevt.py:43} INFO - loaded evt Evt(  1,"natural","dayabay","dayabay/natural/  1 : ", seqs="[]") 20160823-1833 /tmp/blyth/opticks/evt/dayabay/natural/1/fdom.npy
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :      (1142140, 4, 4) : (photons) final photon step 
       wl :           (1142140,) : (photons) wavelength 
     post :         (1142140, 4) : (photons) final photon step: position, time 
     dirw :         (1142140, 4) : (photons) final photon step: direction, weight  
     polw :         (1142140, 4) : (photons) final photon step: polarization, wavelength  
    flags :           (1142140,) : (photons) final photon step: flags  
       c4 :           (1142140,) : (photons) final photon step: dtype split uint8 view of ox flags 
    rx_raw :  (1142140, 10, 2, 4) : (records) photon step records RAW:before reshaping 
       rx :  (1142140, 10, 2, 4) : (records) photon step records 
       ph :      (1142140, 1, 2) : (records) photon history flag/material sequence 
       ps :      (1142140, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 3) 
       rs :  (1142140, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 3) 
      rsr :  (1142140, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 3)  
    Evt(  1,"natural","dayabay","dayabay/natural/  1 : ", seqs="[]") 20160823-1833 /tmp/blyth/opticks/evt/dayabay/natural/1/fdom.npy
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :      (1142140, 4, 4) : (photons) final photon step 
       wl :           (1142140,) : (photons) wavelength 
     post :         (1142140, 4) : (photons) final photon step: position, time 
     dirw :         (1142140, 4) : (photons) final photon step: direction, weight  
     polw :         (1142140, 4) : (photons) final photon step: polarization, wavelength  
    flags :           (1142140,) : (photons) final photon step: flags  
       c4 :           (1142140,) : (photons) final photon step: dtype split uint8 view of ox flags 
    rx_raw :  (1142140, 10, 2, 4) : (records) photon step records RAW:before reshaping 
       rx :  (1142140, 10, 2, 4) : (records) photon step records 
       ph :      (1142140, 1, 2) : (records) photon history flag/material sequence 
       ps :      (1142140, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 3) 
       rs :  (1142140, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 3) 
      rsr :  (1142140, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 3) 
                           1:dayabay 
                       3        1.000        1142140       [1 ] MI
                             1142140         1.00 
                           1:dayabay 
                       0        1.000        1142140       [1 ] ?0?
                             1142140         1.00 

    In [9]: evt.rx[0]
    Out[9]: 
    A()sliced
    A([[[  0,   0,   0,   0],
            [  0,   0,   0, 768]],

           [[  0,   0,   0,   0],
            [  0,   0,   0,   0]],


::

    In [15]: from opticks.ana.base import ffs_

    In [16]: MISS  = 0x1 <<  2   # OpticksPhoton.h

    In [17]: ffs_(MISS)
    Out[17]: 3

    In [18]: miss = ffs_(MISS)

    In [19]: miss << 8 
    Out[19]: 768


As the indexing reported, all photons are missing geometry::

    In [21]: evt.rx.shape
    Out[21]: (1142140, 10, 2, 4)

    In [22]: evt.rx[:,0,1,3]
    Out[22]: 
    A()sliced
    A([768, 768, 768, ..., 768, 768, 768], dtype=int16)

    In [23]: np.unique(evt.rx[:,0,1,3])
    Out[23]: 
    A()sliced
    A([768], dtype=int16)


G4StepNPY.cc is assuming all the same type of genstep...::

    085 void G4StepNPY::relabel(int label)
     86 {
     87 
     88 /*
     89 Scintillation and Cerenkov genstep files contain a pre-label of
     90 a signed integer.::   
     91 
     92     In [7]: sts_(1).view(np.int32)[:,0,0]
     93     Out[7]: array([    1,     2,     3, ..., 13896, 13897, 13898], dtype=int32)
     94 
     95     In [8]: stc_(1).view(np.int32)[:,0,0]
     96     Out[8]: array([   -1,    -2,    -3, ..., -7834, -7835, -7836], dtype=int32)
     97 
     98 Having only 2 types of gensteps is too limiting for example 
     99 when generating test photons corresponding to a light source. 
    100 So *G4StepNPY::relabel* rejigs the markers to a enumerated code.  
    101 The genstep index is still available from the photon buffer, and this is 
    102 written into the *Id* of GPU structs.
    103 
    104 */
    105     LOG(info)<<"G4StepNPY::relabel" ;
    106     for(unsigned int i=0 ; i<m_npy->m_ni ; i++ )
    107     {
    108         int code = m_npy->getInt(i,0u,0u);
    109         if(i % 1000 == 0) printf("G4StepNPY::relabel (%u) %d -> %d \n", i, code, label );
    110         m_npy->setInt(i,0u,0u,0u, label);
    111     }
    112 }
      
::

    In [12]: NATURAL           = 0x1 << 15

    In [13]: NATURAL
    Out[13]: 32768


Labelling with NATURAL is not what is needed, need to label CERENKOV or SCINTILLATION as appropriate::

    In [15]: evt.gs[:,0,0].view(np.int32)
    Out[15]: 
    A()sliced
    A([32768, 32768, 32768, ..., 32768, 32768, 32768], dtype=int32)
     


::

    In [5]: a = np.load(os.path.expanduser("/Users/blyth/opticksdata/gensteps/dayabay/cerenkov/1.npy"))

    In [7]: a[:,0,0]
    Out[7]: array([ nan,  nan,  nan, ...,  nan,  nan,  nan], dtype=float32)

    In [8]: a[:,0,0].view(np.int32)
    Out[8]: array([   -1,    -2,    -3, ..., -7834, -7835, -7836], dtype=int32)

    In [9]: b = np.load(os.path.expanduser("/Users/blyth/opticksdata/gensteps/dayabay/scintillation/1.npy"))

    In [10]: b[:,0,0]
    Out[10]: array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32)

    In [11]: b[:,0,0].view(np.int32)
    Out[11]: array([    1,     2,     3, ..., 13896, 13897, 13898], dtype=int32)





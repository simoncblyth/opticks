GUI Photon Selection Failure
==============================

GUI section of photon seqhis/seqmat subsets titled 
is working for some events, but not others
Any selection other than "All" in both seqhis and seqmat makes 
all photons disappear. 

Progess
---------

* note rs(recsel) shape inconsistencies

Background
-----------

* Vague recollection of doing external indexing for cfg4 to 
  prevent having to index on every load. 

* But in recent developments have got indexing on load to work again.

* Index-on-load via CUDA/Thrust is not a long term solution, as clearly 
  indexing immediately after simulation makes much more 
  sense : index once, visualize many times.  


Approach to solving
--------------------

* Implement CPU only indexing so that the pure G4 workflow 
  can simulate and index without GPU assistance

* develop index test, that checks validity of an index both the 
  frequency table and corresponding lookups from it 


Not Working
--------------

The persisted phosel and recsel are all zeroes::

   ggv-;ggv-pmt-test --cfg4 --load


    In [1]: run pmt_test_evt.py
    WARNING:env.numerics.npy.evt:init_index finds too few (ps)phosel uniques : 1
    WARNING:env.numerics.npy.evt:init_index finds too few (rs)recsel uniques : 1
    Evt(-4,"torch","PmtInBox","PmtInBox/torch/-4 : ", seqs="[]")
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :       (500000, 4, 4) : (photons) final photon step 
       wl :            (500000,) : (photons) wavelength 
     post :          (500000, 4) : (photons) final photon step: position, time 
     dirw :          (500000, 4) : (photons) final photon step: direction, weight  
     polw :          (500000, 4) : (photons) final photon step: polarization, wavelength  
    flags :            (500000,) : (photons) final photon step: flags  
       c4 :            (500000,) : (photons) final photon step: dtype split uint8 view of ox flags 
       rx :   (500000, 10, 2, 4) : (records) photon step records 
       ph :       (500000, 1, 2) : (records) photon history flag/material sequence 
       ps :       (500000, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 1) 
       rs :   (250000, 10, 2, 4) : (records) recsel sequence frequency index lookups (uniques 1) 

    In [2]: evt.ps
    Out[2]: 
    A(pstorch,-4,PmtInBox)(photons) phosel sequence frequency index lookups (uniques 1)
    A([[[0, 0, 0, 0]],

           [[0, 0, 0, 0]],

           [[0, 0, 0, 0]],

           ...

    In [3]: evt.rs
    Out[3]: 
    A()sliced
    A([[[[0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0]],

            [[0, 0, 0, 0],
             [0, 0, 0, 0]],

Also the shape looks funny::

       ps :       (500000, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 1) 
       rs :   (250000, 10, 2, 4) : (records) recsel sequence frequency index lookups (uniques 1) 

Actually this was due to python level reshaping, not a problem with the original, now fixed::

    In [1]: run pmt_test_evt.py
    WARNING:env.numerics.npy.evt:init_index finds too few (ps)phosel uniques : 1
    WARNING:env.numerics.npy.evt:init_index finds too few (rs)recsel uniques : 1
    WARNING:env.numerics.npy.evt:init_index finds too few (rsr)reshaped-recsel uniques : 1
    Evt(-4,"torch","PmtInBox","PmtInBox/torch/-4 : ", seqs="[]")
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :       (500000, 4, 4) : (photons) final photon step 
       wl :            (500000,) : (photons) wavelength 
     post :          (500000, 4) : (photons) final photon step: position, time 
     dirw :          (500000, 4) : (photons) final photon step: direction, weight  
     polw :          (500000, 4) : (photons) final photon step: polarization, wavelength  
    flags :            (500000,) : (photons) final photon step: flags  
       c4 :            (500000,) : (photons) final photon step: dtype split uint8 view of ox flags 
       rx :   (500000, 10, 2, 4) : (records) photon step records 
       ph :       (500000, 1, 2) : (records) photon history flag/material sequence 
       ps :       (500000, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 1) 
       rs :   (500000, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 1) 
      rsr :   (500000, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 1) 



Did recent changes in NumpyEvt mess up the recsel shape? Perhaps are matching record shape ?::

     301     NPY<unsigned char>* phosel = NPY<unsigned char>::make(num_photons,1,4); // shape (np,1,4) (formerly initialized to 0)
     302     setPhoselData(phosel);
     303 
     304     NPY<unsigned char>* recsel = NULL ;
     305     if(m_flat)
     306         recsel = NPY<unsigned char>::make(num_records,1,4); // shape (nr,1,4) (formerly initialized to 0) 
     307     else
     308         recsel = NPY<unsigned char>::make(num_photons, m_maxrec,1,4); // shape (nr,1,4) (formerly initialized to 0) 
     309 
     310     setRecselData(recsel);
     311 }



Working
---------

::

   ggv-;ggv-pmt-test --load

   ggv-;ggv-g4gun --dbg --load --target 3153 --optixviz 


Looking at a good index, see the phosel and recsel are not persisted::

    In [4]: run g4gun.py
    WARNING:env.numerics.npy.evt:init_index finds too few (ps)phosel uniques : 0
    WARNING:env.numerics.npy.evt:init_index finds too few (rs)recsel uniques : 0
    Evt(-1,"G4Gun","G4Gun","G4Gun/G4Gun/-1 : ", seqs="[]")
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :       (226486, 4, 4) : (photons) final photon step 
       wl :            (226486,) : (photons) wavelength 
     post :          (226486, 4) : (photons) final photon step: position, time 
     dirw :          (226486, 4) : (photons) final photon step: direction, weight  
     polw :          (226486, 4) : (photons) final photon step: polarization, wavelength  
    flags :            (226486,) : (photons) final photon step: flags  
       c4 :            (226486,) : (photons) final photon step: dtype split uint8 view of ox flags 
       rx :   (226486, 10, 2, 4) : (records) photon step records 
       ph :       (226486, 1, 2) : (records) photon history flag/material sequence 
       ps :            (0, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 0) 
       rs :        (0, 10, 2, 4) : (records) recsel sequence frequency index lookups (uniques 0) 

    ### rs shape? should it not be  (0, 10, 1, 4) 

Yep, twas incorrect python level reshaping::

       rx :   (226486, 10, 2, 4) : (records) photon step records 
       ph :       (226486, 1, 2) : (records) photon history flag/material sequence 
       ps :            (0, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 0) 
       rs :        (0, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 0) 
       rsr :       (0, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 0) 


::

    In [8]: map(hex_, np.unique(evt.seqmat))
    Out[8]: 
    ['0x11',
     '0x111',
     '0x1111',
     '0x11111',
     '0x111111',
     '0x1111111',
     '0x11111111',
     '0x111111111',
     '0x1111111111']

    In [9]: map(hex_, np.unique(evt.seqhis))
    Out[9]: 
    ['0x4f',
     '0x46f',
     '0x4cf',
     '0x40cf',
     '0x466f',
     '0x4c6f',

    In [13]: print evt.history.table
                            -1:G4Gun 
                      4f        0.927         209840       [2 ] G4GUN AB
              cccbcccccf        0.037           8474       [10] G4GUN BT BT BT BT BT BR BT BT BT
               4cccccccf        0.017           3752       [9 ] G4GUN BT BT BT BT BT BT BT AB
                    4ccf        0.004            884       [4 ] G4GUN BT BT AB
              4ccbcccccf        0.001            299       [10] G4GUN BT BT BT BT BT BR BT BT AB
                b00cc0cf        0.001            261       [8 ] G4GUN BT ?0? BT BT ?0? ?0? BR
              cccccccccf        0.001            261       [10] G4GUN BT BT BT BT BT BT BT BT BT
                 4cccccf        0.001            200       [7 ] G4GUN BT BT BT BT BT AB
                  4ccccf        0.001            195       [6 ] G4GUN BT BT BT BT AB
              4cc00cc0cf        0.001            184       [10] G4GUN BT ?0? BT BT ?0? ?0? BT BT AB
              ccbccccccf        0.001            181       [10] G4GUN BT BT BT BT BT BT BR BT BT
               4cbcccccf        0.001            181       [9 ] G4GUN BT BT BT BT BT BR BT AB
              ccbcccc0cf        0.001            165       [10] G4GUN BT ?0? BT BT BT BT BR BT BT
              4ccccccc6f        0.001            118       [10] G4GUN SC BT BT BT BT BT BT BT AB
              4cccccc0cf        0.001            116       [10] G4GUN BT ?0? BT BT BT BT BT BT AB
                4cc0cccf        0.000             78       [8 ] G4GUN BT BT BT ?0? BT BT AB



compressed-record-into-new-workflow
=====================================

integrate compressed records srec.h into QEvent/qevent
----------------------------------------------------------

* how to configure full or compressed or both  record/rec ? DONE in SEventConfig with _RECORD and _REC  
* where to allocate ?
  
  * QEvent::setNumPhoton allocates photons and records when they are configured in SEventConfig 


Check the compressed rec in CXRaindropTest::

    119     NP* gs = SEvent::MakeTorchGensteps();
    120     cx.setGensteps(gs);  // HMM: passing thru to QEvent, perhaps should directly talk to QEvent ? 
    121     cx.simulate();




domain compression
----------------------

1. domain compression requires domains: 

   * center_extent, time_domain, wavelength_domain :  ce/td/wd


encapsulated domain compressed record : in sysrap/srec.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm how to do this compression more simply and cleanly and more self-contained ?


* design a dedicated compressed record type to live within squad.h 
  that unions with short4 and has compression and decompression methods 

  * needs to be testable on CPU but should use CUDA intrinsics on device 

  * DONE : implemented in sysrap/srec.h see also sysrap/tests/srec_test.cc


old way domain compression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ocu/photon.h::

    029 
    030 #define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
    031 


    108 /**
    109 shortnorm
    110 ------------
    111 
    112 range of short is -32768 to 32767
    113 Expect no positions out of range, as constrained by the geometry are bouncing on,
    114 but getting times beyond the range eg 0.:100 ns is expected
    115 
    116 **/
    117 
    118 PHOTON_METHOD short shortnorm( float v, float center, float extent )
    119 {
    120     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    121     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
    122 }
    123 



    // short4  half of float4 : and are stuffing the record into 2*short4  
    // thats a factor four smaller than the uncompressed photon

    In [2]: np.int16(0xffff)
    Out[2]: -1

    In [3]: np.int16(0xfffe)
    Out[3]: -2

    In [4]: np.int16(0x7fff)
    Out[4]: 32767


::

    epsilon:opticks blyth$ opticks-f qquad
    ./cfg4/CWriter.cc:    qquad qaux ; 
    ./cfg4/CRecorder.h:union CFG4_API qquad
    ./optixrap/cu/photon.h:    qquad qpolw ;    
    ./optixrap/cu/photon.h:    qquad qaux ;  
    ./optixrap/cu/quad.h:union qquad
    ./npy/RecordsNPY.cpp:    124     qquad qpolw ;
    epsilon:opticks blyth$ 

ocu/quad.h::

     33 // "half" sized vector types, all 4*16 = 64 bit       (8 bytes)
     34 union hquad
     35 {
     36    short4   short_ ;
     37    ushort4  ushort_ ;
     38 };
     39 
     40 
     41 // "quarter" sized vector types, all 4*8 = 32 bit   (4 bytes)
     42 union qquad
     43 {
     44    char4   char_   ;
     45    uchar4  uchar_  ;
     46 };






::

    160 // optix::buffer<short4>& rbuffer
    161 
    162 PHOTON_METHOD void rsave( Photon& p, unsigned s_flag, uint4& s_index, short4* rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain, float4& boundary_domain )
    163 {
    164     rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
    165                     shortnorm(p.position.x, center_extent.x, center_extent.w),
    166                     shortnorm(p.position.y, center_extent.y, center_extent.w),
    167                     shortnorm(p.position.z, center_extent.z, center_extent.w),
    168                     shortnorm(p.time      , time_domain.x  , time_domain.y  )
    169                     );
    170 
    171     float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 
    172 
    173     qquad qpolw ;
    174     qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;  // pol : -1->1  pol+1 : 0->2   (pol+1)*127 : 0->254
    175     qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    176     qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    177     qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;
    178 
    179     // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 


    180     hquad polw ;    // union of short4, ushort4
    181     polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    182     polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;



* https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html

::

    __device__ unsigned int __float2uint_rn ( float  x )
        Convert a float to an unsigned integer in round-to-nearest-even mode. 


::

    183 
    184 
    185 #ifdef IDENTITY_CHECK
    186     // spread uint32 photon_id across two uint16
    187     unsigned int photon_id = p.flags.u.y ;
    188     polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    189     polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    190     // OSX intel + CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
    191 #endif
    192      // boundary int and m1 index uint are known to be within char/uchar ranges 
    193     //  uchar: 0 to 255,   char: -128 to 127 
    194     
    195     qquad qaux ;
    196     qaux.uchar_.x =  s_index.x ;    // m1  
    197     qaux.uchar_.y =  s_index.y ;    // m2   
    198     qaux.char_.z  =  p.flags.i.x ;  // boundary(range -55:55)   debugging some funny material codes
    199     qaux.uchar_.w = __ffs(s_flag) ; // first set bit __ffs(0) = 0, otherwise 1->32 
    200     
    201     //             lsb_ (flq[0].x)    msb_ (flq[0].y)
    202     //            
    203     polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    204     
    205     //              lsb_ (flq[0].z)    msb_ (flq[0].w)
    206     polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;
    207     
    208     
    209     rbuffer[record_offset+1] = polw.short_ ;
    210 }


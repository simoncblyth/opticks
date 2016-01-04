#pragma once

// see npy-/numutil.cpp
// http://stackoverflow.com/questions/7337526/how-to-tell-if-a-32-bit-int-can-fit-in-a-16-bit-short
// http://en.wikipedia.org/wiki/Two's_complement
// http://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers
// http://stereopsis.com/radix.html
// https://graphics.stanford.edu/~seander/bithacks.html
// http://www.informit.com/articles/article.aspx?p=2033340&seqNum=3

#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)

#include "OpticksPhoton.h"
#include "OpticksFlags.h"

enum { BREAK, CONTINUE, PASS, START, RETURN }; // return value from propagate_to_boundary


#ifdef __CUDACC__


struct Photon
{
   float3 position ;  
   float  time ;  

   float3 direction ;
   float  weight ; 

   float3 polarization ;
   float  wavelength ; 

   quad flags ;     // x:boundary  y:photon_id   z:m1   w:history 
                    //             [debug-only]
};



__device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset)
{
    pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time ); 
    pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
    pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
    pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w); 
}






__device__ short shortnorm( float v, float center, float extent )
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //
    int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 


__device__ void rsave( Photon& p, State& s, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
{
    //  pack position and time into normalized shorts (4*16 = 64 bits)
    //
    //  TODO: use a more vectorized approach, ie
    // 
    //  * combine position and time domains into single float4 on the host 
    //  * after verification can dispense with the fit checking for positions, just do time
    //        
    //  * adopt p.position_time  maybe p.polarization_wavelength
    //  * simularly with domains of those ?
    // 
    rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
                    shortnorm(p.position.x, center_extent.x, center_extent.w), 
                    shortnorm(p.position.y, center_extent.y, center_extent.w), 
                    shortnorm(p.position.z, center_extent.z, center_extent.w),   
                    shortnorm(p.time      , time_domain.x  , time_domain.y  )
                    ); 

    //    plt.hist(a[::10,0,3]/100., bins=100)    first record times in range 0:45 ns with 100ns domain
    //
    //  pack polarization and wavelength into 4*8 = 32 bits   
    //  range of char -128:127  normalization of polarization and wavelength expected bulletproof, so no handling of out-of-range 
    //  range of uchar 0:255   -1.f:1.f  + 1 => 0.f:2.f  so scale by 127.f 
    //
    //  polarization already normalized into -1.f:1.f
    //  wavelenth normalized via  (wavelength - low)/range into 0.:1. 
    //
    // TODO: avoid reliance on wavelength_lookup here ?

    float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 

    qquad qpolw ;    
    qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;
    qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;
    qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;

    // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 
    hquad polw ;    // union of short4, ushort4
    polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;
    

#ifdef IDENTITY_CHECK
    // spread uint32 photon_id across two uint16
    unsigned int photon_id = p.flags.u.y ;
    polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    // OSX intel + CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
#endif


     // boundary int and m1 index uint are known to be within char/uchar ranges 
    //  uchar: 0 to 255,   char: -128 to 127 

    qquad qaux ;  
    qaux.uchar_.x =  s.index.x ;    // m1  
    qaux.uchar_.y =  s.index.y ;    // m2   
    qaux.char_.z  =  p.flags.i.x ;  // boundary(range -55:55)   debugging some funny material codes
    qaux.uchar_.w = __ffs(s.flag) ; // first set bit __ffs(0) = 0, otherwise 1->32 

    // DEBUG: non-reproducibility, 
    //        setting these to zero avoids the 1 in 2000 level non-reproducibility in "ggv --noindex" 
    //        interop running
    //
    //qaux.uchar_.x = 0 ;
    //qaux.uchar_.y = 0 ;
   
  
    //             lsb_ (flq[0].x)    msb_ (flq[0].y)
    //            
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;   
 
    //              lsb_ (flq[0].z)    msb_ (flq[0].w)
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;


    rbuffer[record_offset+1] = polw.short_ ; 

}


#endif


/*


Observe non-reproducibility in interop running 
at level of 1 in 2000 of entry [:,1,2]



n [35]: d10 = a[:,1,0] != b[:,1,0]

In [36]: d10.size
Out[36]: 6128410

In [37]: a[:,1,0][d10].size
Out[37]: 0

In [38]: d11 = a[:,1,1] != b[:,1,1]

In [39]: a[:,1,1][d11].size
Out[39]: 0

In [40]: d12 = a[:,1,2] != b[:,1,2]

In [41]: a[:,1,2][d12].size
Out[41]: 25464

In [42]: d13 = a[:,1,3] != b[:,1,3]

In [43]: a[:,1,3][d13].size
Out[43]: 0



Non-reproducibility on consequtive running of m2,
no-such issue seen with Juno



In [11]: a1 = np.array( a[:,1,2].view(np.uint16) & 0xFF , dtype=np.uint8 )

In [12]: a1
Out[12]: array([10, 10, 11, ...,  0,  0,  0], dtype=uint8)

In [13]: b1 = np.array( b[:,1,2].view(np.uint16) & 0xFF , dtype=np.uint8 )

In [14]: b1
Out[14]: array([10, 10, 11, ...,  0,  0,  0], dtype=uint8)

In [15]: d1 = a1 != b1

In [16]: a1[d1]
Out[16]: array([], dtype=uint8)

In [17]: b1[d1]
Out[17]: array([], dtype=uint8)

In [18]: a2 = np.array( a[:,1,2].view(np.uint16) >> 8 , dtype=np.uint8 )

In [19]: b2 = np.array( b[:,1,2].view(np.uint16) >> 8 , dtype=np.uint8 )

In [20]: a2
Out[20]: array([ 10,  11,  12, ..., 128, 128, 128], dtype=uint8)

In [21]: b2
Out[21]: array([ 10,  11,  12, ..., 128, 128, 128], dtype=uint8)

In [22]: d2 = a2 != b2

In [23]: a2[d2].size
Out[23]: 25464

In [24]: a2[d2]
Out[24]: array([10, 10, 10, ...,  1, 18, 10], dtype=uint8)

In [25]: b2[d2]
Out[25]: array([16, 18, 18, ..., 10, 10, 18], dtype=uint8)


In [31]: count_unique(b2[d2])
Out[31]: 
array([[    1,   705],
       [    2,    11],
       [   10,  8214],
       [   11,  2289],
       [   12,   139],
       [   16,  2032],
       [   18, 11017],
       [   20,   406],
       [   21,   592],
       [   24,    59]])

In [32]: count_unique(a2[d2])
Out[32]: 
array([[    1,   789],
       [    2,    18],
       [   10,  8182],
       [   11,  2154],
       [   12,   176],
       [   16,  1786],
       [   18, 11372],
       [   20,   343],
       [   21,   609],
       [   24,    35]])


{
    "ADTableStainlessSteel": "19",
    "Acrylic": "3",
    "Air": "15",
    "Aluminium": "18",
    "Bialkali": "5",
    "DeadWater": "8",
    "ESR": "10",
    "Foam": "20",
    "GdDopedLS": "1",
    "IwsWater": "6",
    "LiquidScintillator": "2",
    "MineralOil": "4",
    "Nitrogen": "21",
    "NitrogenGas": "22",
    "Nylon": "23",
    "OwsWater": "9",
    "PPE": "17",
    "PVC": "24",
    "Pyrex": "14",
    "Rock": "16",
    "StainlessSteel": "12",
    "Tyvek": "25",
    "UnstStainlessSteel": "11",
    "Vacuum": "13",
    "Water": "7"
}













*/





/*


In [68]: a.shape
Out[68]: (6128410, 2, 4)      // 8 * int16  item

In [69]: a.dtype
Out[69]: dtype('int16')

    //  f = msb2_(a.view(np.uint16)[:,1,3])

In [9]: count_unique(f)
Out[9]: 
array([[      1,  559369],
       [      3,   59502],
       [      4,  469483],
       [      5,  224054],
       [      6,   13711],
       [     11,   62549],
       [     12, 1204949],
       [    128, 3534793]])





::

    In [1]: r = rxc_(1)
    INFO:env.g4dae.types:loading /usr/local/env/rxcerenkov/1.npy 
    -rw-r--r--  1 blyth  staff  98054640 Jul  8 11:51 /usr/local/env/rxcerenkov/1.npy


    In [12]: m1 = np.array( r[:,1,2].view(np.uint16) & 0xFF , dtype=np.uint8 )  # little endian lsb first

    In [13]: m1
    Out[13]: array([ 6,  6, 12, ...,  0,  0,  0], dtype=uint8)

    In [14]: count_unique(m1)    # hmm iregularity with 0 vs 128  between m1 and m2 
    Out[14]: 
    array([[      0, 3594296],
           [      1,  687092],
           [      2,  385457],
           [      3,  686761],
           [      4,  501392],
           [      5,     631],
           [      6,   60230],
           [      7,    1590],
           [     10,   23636],
           [     11,    8330],
           [     12,  115360],
           [     13,   25966],
           [     14,   28008],
           [     15,    4931],
           [     16,    4718],
           [     21,       7],
           [     24,       5]])


    In [9]: m2 = np.array( r[:,1,2].view(np.uint16) >> 8 , dtype=np.uint8 )

    In [10]: m2
    Out[10]: array([  6,  12,   4, ..., 128, 128, 128], dtype=uint8)

    In [11]: count_unique(m2)
    Out[11]: 
    array([[      1,   27873],
           [      2,  298400],
           [      3, 1397602],
           [      4,  338262],
           [      5,   12556],
           [      6,  157076],
           [      7,     735],
           [     10,    4847],
           [     11,   74522],
           [     12,  156963],
           [     13,   26474],
           [     14,   27211],
           [     15,   58411],
           [     16,   11312],
           [     21,    1171],
           [     24,     192],
           [    128, 3534803]])


    In [21]: q3 = np.array( r[:,1,3].view(np.uint16) & 0xFF , dtype=np.uint8 )  # little endian lsb first 

    In [22]: count_unique(q3)
    Out[22]: array([[      0, 6128410]])


    In [18]: q4 = np.array( r[:,1,3].view(np.uint16) >> 8 , dtype=np.uint8 )

    In [19]: q4
    Out[19]: array([  1,  12,  12, ..., 128, 128, 128], dtype=uint8)

    In [20]: count_unique(q4)
    Out[20]: 
    array([[      1,  559369],
           [      3,   59493],
           [      4,  453843],
           [      5,  224211],
           [      6,   13762],
           [     11,   65044],
           [     12, 1217885],
           [    128, 3534803]])






*/




//  Correspondence to gl/rec/geom.glsl 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  * NumpyEvt::setRecordData sets rflq buffer input as ViewNPY::BYTE starting from offset 2 (ie .z) 
//                                                                                                     
//      flq[0].x   <->  lowbyte  polw.ushort_.z    <->  polw.ushort_.z & 0x00FF            m1 = np.array( a[:,1,2] >> 8, dtype=np.int8 )
//
//      flq[0].y   <->  highbyte polw.ushort_.z    <->  polw.ushort_.z & 0xFF00 >> 8 
//
//  This follows from little endianness (lesser numerical significance at lower address) 
//  of CUDA GPUs : so when interpreting a ushort as two uchar (as OpenGL is doing) 
//  the low byte comes first, opposite to the "writing" order 
//
//                             // flq[0].x m1 colors 
// polw.ushort_.z = 0x0e0c ;   // green(0c) 
// polw.ushort_.z = 0x0c0e ;   //  cyan(0e)
// polw.ushort_.z = 0x0c0f ;   //   red(0f) 
//
//

/*

   (m1/bd check)  after move to local material and surface indices

::

    In [1]: a = rxc_(1)

    In [2]: m1 = np.array( a[:,1,2] >> 8, dtype=np.int8 )  ## all records, 2nd quad, z, high char

    In [3]: m1
    Out[3]: array([  10,   10,   11, ..., -128, -128, -128], dtype=int8)

    In [4]: m1[:1000].reshape(-1,10)
    Out[4]: 
    array([[  10,   10,   11,   11, -128, -128, -128, -128, -128, -128],
           [  10,   10,   10,   10, -128, -128, -128, -128, -128, -128],
           [  10,   10, -128, -128, -128, -128, -128, -128, -128, -128],
           [  10,   10,   10,   10, -128, -128, -128, -128, -128, -128],
           [  10,   10,   10,   10, -128, -128, -128, -128, -128, -128],
           [  10,   10,   10, -128, -128, -128, -128, -128, -128, -128],
           [  10,   10,   10, -128, -128, -128, -128, -128, -128, -128],


In [5]: m1[m1>0]
Out[5]: array([10, 10, 11, ..., 10, 10, 10], dtype=int8)

In [6]: count_unique(m1[m1>0])
Out[6]: 
array([[     1,  23905],
       [     2,   6745],
       [     3,   7588],
       [    10,  66096],
       [    11, 127872],
       [    12, 483114],
       [    13, 690070],
       [    14, 383238],
       [    15, 687130],
       [    16,  27962],
       [    17,    584],
       [    18,   8804],
       [    19,  25401],
       [    20,   1623],
       [    21,      6],
       [    24,      7]])


 name                      Vacuum source         76 local           1
 name                        Rock source         70 local           2
 name                         Air source         44 local           3
 name                         PPE source         66 local           4
 name                   Aluminium source         45 local           5
 name                        Foam source         53 local           6
 name                   DeadWater source         51 local           7
 name                       Tyvek source         74 local           8
 name                    OwsWater source         65 local           9
 name                    IwsWater source         57 local          10
 name              StainlessSteel source         72 local          11
 name                  MineralOil source         59 local          12
 name                     Acrylic source         43 local          13
 name          LiquidScintillator source         58 local          14
 name                   GdDopedLS source         54 local          15
 name                       Pyrex source         68 local          16
 name                    Bialkali source         48 local          17
 name          UnstStainlessSteel source         75 local          18
 name                         ESR source         52 local          19
 name                       Water source         77 local          20
 name                    Nitrogen source         61 local          21
 name                       Nylon source         63 local          22
 name                 NitrogenGas source         62 local          23
 name                         PVC source         67 local          24
 name       ADTableStainlessSteel source         42 local          25














   bd = np.array( a[:,1,2] & 0xFF , dtype=np.int8 )   # huh these are m1

In [20]: bd[100000:100000+100].reshape(-1,10)
Out[20]: 
array([[58, 58, 43, 59, 43, 59, 59, 43, 59, 43],
       [43, 58, 43, 59, 43, 44, 52, 52,  0,  0],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0],
       [58, 58, 58, 58, 43, 54, 43, 58, 43, 59],
       [58, 43, 59, 43, 59, 59, 43, 59, 43, 58],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0],
       [58, 58,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int8)






    In [50]: bd = np.array( a[:,1,2] & 0xFF , dtype=np.int8 )

    In [51]: bd
    Out[51]: array([-12, -13, -14, ...,   0,   0,   0], dtype=int8)

    In [52]: bd[:1000].reshape(-1,10)
    Out[52]: 
    array([[-12, -13, -14, -14,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -12,   0,   0,   0,   0,   0,   0,   0,   0],
           [-12, -13,  12, -12, -12,   0,   0,   0,   0,   0],
           [-12, -13, -13,   0,   0,   0,   0,   0,   0,   0],





   (identity check, depending on same endianness of host and device)

::

    In [1]: r = rxc_(1)

    In [2]: (r.shape, r.dtype)
    Out[2]: ((6128410, 2, 4), dtype('int16'))

    In [7]: r.view(np.uint16).view(np.uint32)[::10,1,1]   # scrunch up 2 uint16 into a uint32
    Out[7]: array([     0,      1,      2, ..., 612838, 612839, 612840], dtype=uint32)

    In [8]: np.all(np.arange(0,612841,dtype=np.uint32) == r.view(np.uint32)[::10,1,1])  ## chaining views dont make much sense
    Out[8]: True


    ## alternatively without the dirty scrunching 

    In [30]: lss = np.array( r.view(np.uint16)[::10,1,2], dtype=np.uint32 )  ## flags.z

    In [31]: mss = np.array( r.view(np.uint16)[::10,1,3], dtype=np.uint32 )  ## flags.w

    In [32]: mss << 16 | lss 
    Out[32]: array([     0,      1,      2, ..., 612838, 612839, 612840], dtype=uint32)

    In [33]: np.all( (mss << 16) | lss == np.arange(0,612841, dtype=np.uint32 ))
    Out[33]: True



    (lightly packed) 

    check packing results using ::MAXREC to pick generated 0th slot only  

    In [1]: a = rxc_(1)

       plt.hist(a[::10,1,0], range=(-128,127), bins=256 )  # bins+1 edges 
       plt.hist(a[::10,1,1], range=(-128,127), bins=256 )  # bins+1 edges 
       plt.hist(a[::10,1,2], range=(-128,127), bins=256 )  # bins+1 edges 
  
       plt.hist(a[::10,1,3], range=(-128,127), bins=256 )  # bins+1 edges 


    (tightly packed, using signed normalization)  

     hmm get no negatives, for tight packing better to use unsigned 

       plt.hist( a[::10,1,0] & 0xFF , bins=256, range=(-128,127) ) 


    (tightly packed, using unsigned normalization)

       plt.hist( a[::10,1,0] & 0xFF , bins=256, range=(0,256) )
       plt.hist( a[::10,1,0] >> 8   , bins=256, range=(0,256) )
       plt.hist( a[::10,1,1] & 0xFF , bins=256, range=(0,256) )

       plt.hist( a[::10,1,1] >> 8   , bins=256, range=(0,256) )  ## oops runs into sign bit yields halfed and folded

       plt.hist( a.view(np.uint16)[::10,1,1] >> 8, bins=256, range=(0,255) )   ## zero bin lower... due to bad number of bins
       plt.hist( a.view(np.uint16)[::10,1,1] >> 8, bins=255, range=(0,255) )   ## better mpl provides bins+1 edges
                                               
       wavelength is occupyung the MSB, so runs into the sign bit 
       hence its necessary to view(np.uint16)
       its more correct to do that for all when using unorm but only matters for MSB



*/




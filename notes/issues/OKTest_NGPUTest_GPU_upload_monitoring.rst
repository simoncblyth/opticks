OKTest_NGPUTest_GPU_upload_monitoring
========================================

Overview
-----------

* implemented NGPU for recording of GPU uploads, see NGPU.hpp for details
* no smoking guns, but lots of things to follow up, tidy up and improve 


OKTest --gpumon
-----------------

Observations

OptiX triangulated 
~~~~~~~~~~~~~~~~~~~~~

OGeo0-0.  
    identity/vertex/index  (no lod for global) 
  
OGeo2-0.  non-lod
OGeo2-1.  lod
    
    * OGeo2/3/4/5  
    * identity/vertex/index

    * mm5 triangulated identity is 31481856 31.48 MB 
      (does that make sense?)

    * non-lod and lod are same size, so lod is doubling usage !!!

::

    NGPUTest $TMP/OKTest_GPUMon.npy   > $TMP/OKTest_GPUMon.txt
    NGPUTest $TMP/OKX4Test_GPUMon.npy > $TMP/OKX4Test_GPUMon.txt

    diff --width 200 -y OKX4Test_GPUMon.txt OKTest_GPUMon.txt


::

    epsilon:opticks blyth$ gpumon.py 
            86 :       103381296 : 103.38 : $TMP/OTracerTest_GPUMon.npy 
           127 :        95859616 :  95.86 : $TMP/OKX4Test_GPUMon.npy 
                                131 :     $TMP/OTracerTest_GPUMon.npy  :        $TMP/OKX4Test_GPUMon.npy   
              itransfo/nrm/RBuf:upl :                   64 :   0.00    :                   64 :   0.00     
              vertices/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
                colors/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
               normals/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
               indices/nrm/RBuf:upl :              4844544 :   4.84    :              5511936 :   5.51     
           itransfo/nrmvec/RBuf:upl :                   64 :   0.00    :                   64 :   0.00     
           vertices/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
             colors/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
            normals/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
            indices/nrmvec/RBuf:upl :              4844544 :   4.84    :              5511936 :   5.51     
            itransfo/inrm0/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
            vertices/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
              colors/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             normals/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             indices/inrm0/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
          itransfo/inrm0bb/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
          vertices/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
            colors/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           normals/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           indices/inrm0bb/RBuf:upl :                  144 :   0.00    :                  144 : 

::

    epsilon:sysrap blyth$ gpumon.py 
            94 :       131781536 : 131.78 : $TMP/OKTest_GPUMon.npy 
           127 :        95859616 :  95.86 : $TMP/OKX4Test_GPUMon.npy 
    139
              itransfo/nrm/RBuf:upl :                   64 :   0.00    :                   64 :   0.00     
              vertices/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
                colors/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
               normals/nrm/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
               indices/nrm/RBuf:upl :              4844544 :   4.84    :              5511936 :   5.51     
           itransfo/nrmvec/RBuf:upl :                   64 :   0.00    :                   64 :   0.00     
           vertices/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
             colors/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
            normals/nrmvec/RBuf:upl :              2453568 :   2.45    :              2800512 :   2.80     
            indices/nrmvec/RBuf:upl :              4844544 :   4.84    :              5511936 :   5.51     
            itransfo/inrm0/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
            vertices/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
              colors/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             normals/inrm0/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             indices/inrm0/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
          itransfo/inrm0bb/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
          vertices/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
            colors/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           normals/inrm0bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           indices/inrm0bb/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
            itransfo/inrm1/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
            vertices/inrm1/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
              colors/inrm1/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             normals/inrm1/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             indices/inrm1/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
          itransfo/inrm1bb/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
          vertices/inrm1bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
            colors/inrm1bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           normals/inrm1bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           indices/inrm1bb/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
            itransfo/inrm2/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
            vertices/inrm2/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
              colors/inrm2/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             normals/inrm2/RBuf:upl :                   96 :   0.00    :                   96 :   0.00     
             indices/inrm2/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
          itransfo/inrm2bb/RBuf:upl :                55296 :   0.06    :                55296 :   0.06     
          vertices/inrm2bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
            colors/inrm2bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           normals/inrm2bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           indices/inrm2bb/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
            itransfo/inrm3/RBuf:upl :                43008 :   0.04    :                43008 :   0.04     
            vertices/inrm3/RBuf:upl :                17688 :   0.02    :                17976 :   0.02     
              colors/inrm3/RBuf:upl :                17688 :   0.02    :                17976 :   0.02     
             normals/inrm3/RBuf:upl :                17688 :   0.02    :                17976 :   0.02     
             indices/inrm3/RBuf:upl :                35136 :   0.04    :                35712 :   0.04     
          itransfo/inrm3bb/RBuf:upl :                43008 :   0.04    :                43008 :   0.04     
          vertices/inrm3bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
            colors/inrm3bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           normals/inrm3bb/RBuf:upl :                  288 :   0.00    :                  288 :   0.00     
           indices/inrm3bb/RBuf:upl :                  144 :   0.00    :                  144 :   0.00     
              colors/OColors/OScene :                 1024 :   0.00    :                                   
           OSourceL/OPropLib/OScene :                 4096 :   0.00    :                 4096 :   0.00     
           OScintil/OPropLib/OScene :                16384 :   0.02    :                16384 :   0.02     
           primBuff/OGeo0-0/ciubNPY :                                  :                49856 :   0.05     
           partBuff/OGeo0-0/ciubNPY :                                  :               766976 :   0.77     
           identity/OGeo0-0/cibGBuf :              6459392 :   6.46    :                                   
           tranBuff/OGeo0-0/ciubNPY :                                  :              1026048 :   1.03     
           vertexBu/OGeo0-0/cibGBuf :              2453568 :   2.45    :                                   
            identity/OGeo0-0/cibNPY :                                  :                   16 :   0.00     
           indexBuf/OGeo0-0/cibGBuf :              4844544 :   4.84    :                                   
           planBuff/OGeo0-0/ciubNPY :                                  :                10752 :   0.01     
           identity/OGeo2-0/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
           vertexBu/OGeo2-0/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo2-0/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
           identity/OGeo2-1/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
           vertexBu/OGeo2-1/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo2-1/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
           primBuff/OGeo2-0/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo2-0/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo2-0/ciubNPY :                                  :                  192 :   0.00     
            identity/OGeo2-0/cibNPY :                                  :                13824 :   0.01     
           planBuff/OGeo2-0/ciubNPY :                                  :                    0 :   0.00     
           primBuff/OGeo2-1/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo2-1/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo2-1/ciubNPY :                                  :                  192 :   0.00     
            identity/OGeo2-1/cibNPY :                                  :                13824 :   0.01     
           planBuff/OGeo2-1/ciubNPY :                                  :                    0 :   0.00     
           identity/OGeo3-0/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
           vertexBu/OGeo3-0/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo3-0/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
           identity/OGeo3-1/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
           vertexBu/OGeo3-1/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo3-1/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
           primBuff/OGeo3-0/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo3-0/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo3-0/ciubNPY :                                  :                  192 :   0.00     
            identity/OGeo3-0/cibNPY :                                  :                13824 :   0.01     
           planBuff/OGeo3-0/ciubNPY :                                  :                    0 :   0.00     
           primBuff/OGeo3-1/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo3-1/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo3-1/ciubNPY :                                  :                  192 :   0.00     
              vpos/axis_att/Rdr:upl :                  144 :   0.00    :                                   
            identity/OGeo3-1/cibNPY :                                  :                13824 :   0.01     
              vpos/genstep_/Rdr:upl :                   96 :   0.00    :                                   
              vpos/nopstep_/Rdr:upl :                    0 :   0.00    :                                   
           planBuff/OGeo3-1/ciubNPY :                                  :                    0 :   0.00     
           identity/OGeo4-0/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
              vpos/photon_a/Rdr:upl :              6400000 :   6.40    :                                   
              rpos/record_a/Rdr:upl :             16000000 :  16.00    :                                   
           vertexBu/OGeo4-0/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo4-0/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
              phis/sequence/Rdr:upl :              1600000 :   1.60    :                                   
           identity/OGeo4-1/cibGBuf :               165888 :   0.17    :               165888 :   0.17     
              psel/phosel_a/Rdr:upl :               400000 :   0.40    :                                   
              rsel/recsel_a/Rdr:upl :              4000000 :   4.00    :                                   
           vertexBu/OGeo4-1/cibGBuf :                   96 :   0.00    :                   96 :   0.00     
           indexBuf/OGeo4-1/cibGBuf :                  144 :   0.00    :                  144 :   0.00     
           primBuff/OGeo4-0/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo4-0/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo4-0/ciubNPY :                                  :                  192 :   0.00     
            identity/OGeo4-0/cibNPY :                                  :                13824 :   0.01     
           planBuff/OGeo4-0/ciubNPY :                                  :                    0 :   0.00     
           primBuff/OGeo4-1/ciubNPY :                                  :                   16 :   0.00     
           partBuff/OGeo4-1/ciubNPY :                                  :                   64 :   0.00     
           tranBuff/OGeo4-1/ciubNPY :                                  :                  192 :   0.00     
            identity/OGeo4-1/cibNPY :                                  :                13824 :   0.01     
           planBuff/OGeo4-1/ciubNPY :                                  :                    0 :   0.00     
           identity/OGeo5-0/cibGBuf :             31481856 :  31.48    :             31997952 :  32.00     
           vertexBu/OGeo5-0/cibGBuf :                17688 :   0.02    :                17976 :   0.02     
           indexBuf/OGeo5-0/cibGBuf :                35136 :   0.04    :                35712 :   0.04     
           identity/OGeo5-1/cibGBuf :             31481856 :  31.48    :             31997952 :  32.00     
           vertexBu/OGeo5-1/cibGBuf :                17688 :   0.02    :                17976 :   0.02     
           indexBuf/OGeo5-1/cibGBuf :                35136 :   0.04    :                35712 :   0.04     
           primBuff/OGeo5-0/ciubNPY :                                  :                   80 :   0.00     
           partBuff/OGeo5-0/ciubNPY :                                  :                 2624 :   0.00     
           tranBuff/OGeo5-0/ciubNPY :                                  :                 2304 :   0.00     
            identity/OGeo5-0/cibNPY :                                  :                10752 :   0.01     
           planBuff/OGeo5-0/ciubNPY :                                  :                    0 :   0.00     
           primBuff/OGeo5-1/ciubNPY :                                  :                   80 :   0.00     
           partBuff/OGeo5-1/ciubNPY :                                  :                 2624 :   0.00     
           tranBuff/OGeo5-1/ciubNPY :                                  :                 2304 :   0.00     
            identity/OGeo5-1/cibNPY :                                  :                10752 :   0.01     
           planBuff/OGeo5-1/ciubNPY :                                  :                    0 :   0.00     
            OBndLib/OPropLib/OScene :               614016 :   0.61    :               429312 :   0.43     
              vertices/tex/RBuf:upl :                   48 :   0.00    :                   48 :   0.00     
                colors/tex/RBuf:upl :                   48 :   0.00    :                   48 :   0.00     
               normals/tex/RBuf:upl :                   48 :   0.00    :                   48 :   0.00     
              texcoord/tex/RBuf:upl :                   32 :   0.00    :                   32 :   0.00     
               indices/tex/RBuf:upl :                   24 :   0.00    :                   24 :   0.00     
    epsilon:sysrap blyth$ 





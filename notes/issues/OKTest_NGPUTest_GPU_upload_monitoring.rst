OKTest_NGPUTest_GPU_upload_monitoring
========================================


Overview
-----------

* implemented NGPU for recording of GPU uploads, see NGPU.hpp for details


OKTest --gpumon
-----------------

::

    epsilon:npy blyth$ NGPUTest /tmp/blyth/opticks/GPUMonPath.npy
    2018-07-04 20:11:51.688 INFO  [1256890] [NGPU::dump@114] NGPU::dump num_records 94 num_bytes 131781536
           itransfo       nrm.....       RBuf:upl :              64 :       0.00
           vertices       nrm.....       RBuf:upl :         2453568 :       2.45
           colors..       nrm.....       RBuf:upl :         2453568 :       2.45
           normals.       nrm.....       RBuf:upl :         2453568 :       2.45
           indices.       nrm.....       RBuf:upl :         4844544 :       4.84

           itransfo       nrmvec..       RBuf:upl :              64 :       0.00
           vertices       nrmvec..       RBuf:upl :         2453568 :       2.45
           colors..       nrmvec..       RBuf:upl :         2453568 :       2.45
           normals.       nrmvec..       RBuf:upl :         2453568 :       2.45
           indices.       nrmvec..       RBuf:upl :         4844544 :       4.84

           itransfo       inrm0...       RBuf:upl :           55296 :       0.06
           vertices       inrm0...       RBuf:upl :              96 :       0.00
           colors..       inrm0...       RBuf:upl :              96 :       0.00
           normals.       inrm0...       RBuf:upl :              96 :       0.00
           indices.       inrm0...       RBuf:upl :             144 :       0.00

           itransfo       inrm0bb.       RBuf:upl :           55296 :       0.06
           vertices       inrm0bb.       RBuf:upl :             288 :       0.00
           colors..       inrm0bb.       RBuf:upl :             288 :       0.00
           normals.       inrm0bb.       RBuf:upl :             288 :       0.00
           indices.       inrm0bb.       RBuf:upl :             144 :       0.00

           itransfo       inrm1...       RBuf:upl :           55296 :       0.06
           vertices       inrm1...       RBuf:upl :              96 :       0.00
           colors..       inrm1...       RBuf:upl :              96 :       0.00
           normals.       inrm1...       RBuf:upl :              96 :       0.00
           indices.       inrm1...       RBuf:upl :             144 :       0.00

           itransfo       inrm1bb.       RBuf:upl :           55296 :       0.06
           vertices       inrm1bb.       RBuf:upl :             288 :       0.00
           colors..       inrm1bb.       RBuf:upl :             288 :       0.00
           normals.       inrm1bb.       RBuf:upl :             288 :       0.00
           indices.       inrm1bb.       RBuf:upl :             144 :       0.00

           itransfo       inrm2...       RBuf:upl :           55296 :       0.06
           vertices       inrm2...       RBuf:upl :              96 :       0.00
           colors..       inrm2...       RBuf:upl :              96 :       0.00
           normals.       inrm2...       RBuf:upl :              96 :       0.00
           indices.       inrm2...       RBuf:upl :             144 :       0.00

           itransfo       inrm2bb.       RBuf:upl :           55296 :       0.06
           vertices       inrm2bb.       RBuf:upl :             288 :       0.00
           colors..       inrm2bb.       RBuf:upl :             288 :       0.00
           normals.       inrm2bb.       RBuf:upl :             288 :       0.00
           indices.       inrm2bb.       RBuf:upl :             144 :       0.00

           itransfo       inrm3...       RBuf:upl :           43008 :       0.04
           vertices       inrm3...       RBuf:upl :           17688 :       0.02
           colors..       inrm3...       RBuf:upl :           17688 :       0.02
           normals.       inrm3...       RBuf:upl :           17688 :       0.02
           indices.       inrm3...       RBuf:upl :           35136 :       0.04

           itransfo       inrm3bb.       RBuf:upl :           43008 :       0.04
           vertices       inrm3bb.       RBuf:upl :             288 :       0.00
           colors..       inrm3bb.       RBuf:upl :             288 :       0.00
           normals.       inrm3bb.       RBuf:upl :             288 :       0.00
           indices.       inrm3bb.       RBuf:upl :             144 :       0.00

           colors..       OColors.       OScene.. :            1024 :       0.00
           OSourceL       OPropLib       OScene.. :            4096 :       0.00
           OScintil       OPropLib       OScene.. :           16384 :       0.02

           this is OptiX triangulated geometry for each mm:

           identity       OGeo0-0.       cibGBuf. :         6459392 :       6.46
           vertexBu       OGeo0-0.       cibGBuf. :         2453568 :       2.45
           indexBuf       OGeo0-0.       cibGBuf. :         4844544 :       4.84
           ## no lod for the global 

           identity       OGeo2-0.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo2-0.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo2-0.       cibGBuf. :             144 :       0.00

           identity       OGeo2-1.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo2-1.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo2-1.       cibGBuf. :             144 :       0.00

           identity       OGeo3-0.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo3-0.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo3-0.       cibGBuf. :             144 :       0.00

           identity       OGeo3-1.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo3-1.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo3-1.       cibGBuf. :             144 :       0.00

           identity       OGeo4-0.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo4-0.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo4-0.       cibGBuf. :             144 :       0.00

           identity       OGeo4-1.       cibGBuf. :          165888 :       0.17
           vertexBu       OGeo4-1.       cibGBuf. :              96 :       0.00
           indexBuf       OGeo4-1.       cibGBuf. :             144 :       0.00

           identity       OGeo5-0.       cibGBuf. :        31481856 :      31.48
           vertexBu       OGeo5-0.       cibGBuf. :           17688 :       0.02
           indexBuf       OGeo5-0.       cibGBuf. :           35136 :       0.04

           identity       OGeo5-1.       cibGBuf. :        31481856 :      31.48
           vertexBu       OGeo5-1.       cibGBuf. :           17688 :       0.02
           indexBuf       OGeo5-1.       cibGBuf. :           35136 :       0.04

           #####   huh why OGeo5 (mm5?) so big ?   its as if its the tri for all the PMTs non-instanced ?
           #####   why is the lod "-1." is just as big as the standard "-0." ?


           OBndLib.       OPropLib       OScene.. :          614016 :       0.61
           vertices       tex.....       RBuf:upl :              48 :       0.00
           colors..       tex.....       RBuf:upl :              48 :       0.00
           normals.       tex.....       RBuf:upl :              48 :       0.00
           texcoord       tex.....       RBuf:upl :              32 :       0.00
           indices.       tex.....       RBuf:upl :              24 :       0.00
           vpos....       axis_att       Rdr:upl. :             144 :       0.00
           vpos....       genstep_       Rdr:upl. :              96 :       0.00
           vpos....       nopstep_       Rdr:upl. :               0 :       0.00
           vpos....       photon_a       Rdr:upl. :         6400000 :       6.40
           rpos....       record_a       Rdr:upl. :        16000000 :      16.00
           phis....       sequence       Rdr:upl. :         1600000 :       1.60
           psel....       phosel_a       Rdr:upl. :          400000 :       0.40
           rsel....       recsel_a       Rdr:upl. :         4000000 :       4.00

                       TOTALS in bytes, Mbytes :  :       131781536 :     131.78





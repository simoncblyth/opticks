requirements-for-SURFACE_DETECT
==================================

What enables Opticks to yield SURFACE_DETECT hits ?

* optical buffer needs to have non-zero index for the surface
  and it must have non-zero detect property   


Hmm somehow feels funny for this to be here, not from geometry side::

    epsilon:cu blyth$ grep optical_buffer *.*
    generate.cu:rtBuffer<uint4>                optical_buffer; 
    generate.cu:             slot == 0 ? optical_buffer[MaterialIndex].x : s.index.z, \
    state.h:    s.optical = optical_buffer[su_line] ;   // index/type/finish/value
    state.h:    s.index.x = optical_buffer[m1_line].x ; // m1 index
    state.h:    s.index.y = optical_buffer[m2_line].x ; // m2 index 
    state.h:    s.index.z = optical_buffer[su_line].x ; // su index
    epsilon:cu blyth$ 




    epsilon:optixrap blyth$ grep optical_buffer *.* 
    OBndLib.cc:    optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, nx );
    OBndLib.cc:    memcpy( optical_buffer->map(), obuf->getBytes(), numBytes );
    OBndLib.cc:    optical_buffer->unmap();
    OBndLib.cc:    m_context["optical_buffer"]->setBuffer(optical_buffer);
    OBndLib.hh:2. GBndLib NPY optical buffer into OptiX optical_buffer 
    epsilon:optixrap blyth$ 


    epsilon:GSurfaceLib blyth$ l
    total 80
    -rw-r--r--  1 blyth  staff    400 Jul 19 21:29 GSurfaceLibOptical.npy
    -rw-r--r--  1 blyth  staff   4504 Jul 19 21:29 GPropertyLibMetadata.json
    -rw-r--r--  1 blyth  staff  25040 Jul 19 21:29 GSurfaceLib.npy
    epsilon:GSurfaceLib blyth$ inp *.npy
    a :                                              GSurfaceLib.npy :       (20, 2, 39, 4) : c6bc6dfaa7a9f22cb1c263ab373ac22c : 20200719-2129 
    b :                                       GSurfaceLibOptical.npy :              (20, 4) : 4578b0ffd02acc5b176deb4a9654795e : 20200719-2129 


    In [3]: b
    Out[3]: 
    array([[  0,   0,   3,  20],
           [  1,   0,   0, 100],
           [  2,   0,   1,  99],
           [  3,   0,   0, 100],
           [  4,   0,   0, 100],
           [  5,   0,   1,  99],
           [  6,   0,   0, 100],
           [  7,   0,   0, 100],
           [  8,   0,   0, 100],
           [  9,   0,   0, 100],
           [ 10,   0,   0, 100],
           [ 11,   0,   0, 100],
           [ 12,   0,   1,  99],
           [ 13,   0,   0, 100],
           [ 14,   0,   3,  20],
           [ 15,   0,   3,  20],
           [ 16,   1,   1, 100],
           [ 17,   1,   1, 100],
           [ 18,   1,   1, 100],
           [ 19,   1,   1, 100]], dtype=uint32)



First uint later filled with the index::

    166 guint4 GOpticalSurface::getOptical() const
    167 {
    168    guint4 optical ;
    169    optical.x = UINT_MAX ; //  place holder
    170    optical.y = boost::lexical_cast<unsigned int>(getType());
    171    optical.z = boost::lexical_cast<unsigned int>(getFinish());
    172 
    173    const char* value = getValue();
    174    float percent = boost::lexical_cast<float>(value)*100.f ;   // express as integer percentage 
    175    unsigned upercent = unsigned(percent) ;   // rounds down 
    176 
    177    optical.w = upercent ;
    178 
    179    return optical ;
    180 }

     530 guint4 GSurfaceLib::createOpticalSurface(GPropertyMap<float>* src)
     531 {
     532    assert(src->isSkinSurface() || src->isBorderSurface() || src->isTestSurface());
     533    GOpticalSurface* os = src->getOpticalSurface();
     534    assert(os && "all skin/boundary surface expected to have associated OpticalSurface");
     535    guint4 optical = os->getOptical();
     536    return optical ;
     537 }
     538 
     539 guint4 GSurfaceLib::getOpticalSurface(unsigned int i)
     540 {
     541     GPropertyMap<float>* surf = getSurface(i);
     542     guint4 os = createOpticalSurface(surf);
     543     os.x = i ;
     544     return os ;
     545 }



    105 void OBndLib::convert()
    106 {   
    107     LOG(LEVEL) << "[" ;
    108     
    109     m_blib->createDynamicBuffers();
    110     
    111     NPY<float>* orig = m_blib->getBuffer() ;  // (123, 4, 2, 39, 4)
    112     
    113     assert(orig && "OBndLib::convert orig buffer NULL");
    114     
    115     NPY<float>* buf = m_debug_buffer ? m_debug_buffer : orig ;
    116 
    117     
    118     bool same = buf->hasSameShape(orig) ;
    119     if(!same)
    120         LOG(fatal) << "OBndLib::convert buf/orig shape mismatch "
    121                    << " orig " << orig->getShapeString()
    122                    << " buf " << buf->getShapeString()
    123                    ;
    124     
    125     assert(same);
    126     
    127     makeBoundaryTexture( buf );
    128     
    129     NPY<unsigned int>* obuf = m_blib->getOpticalBuffer() ;  // (123, 4, 4)
    130     
    131     makeBoundaryOptical(obuf);
    132     
    133     if(m_ok->isDbgTex())  // --dbgtex
    134     {   
    135         const char* idpath = m_ok->getIdPath(); 
    136         LOG(fatal) << " --dbgtex saving buf and obuf into " << idpath << "/dbgtex" ;
    137         buf->save(idpath,"dbgtex","buf.npy" ) ; 
    138         obuf->save(idpath,"dbgtex","obuf.npy" ) ;
    139         m_blib->saveNames(idpath, "dbgtex", "bnd.txt");
    140     }
    141     
    142     LOG(LEVEL) << "]" ;
    143 }



    257 void OBndLib::makeBoundaryOptical(NPY<unsigned int>* obuf)
    258 {
    259     unsigned int numBytes = obuf->getNumBytes(0) ;
    260     unsigned int numBnd = numBytes/(GPropertyLib::NUM_MATSUR*4*sizeof(unsigned int)) ;  // this 4 is not NUM_PROP

                                                    4*4*sizeof(unsigned)

    261     unsigned int nx = numBnd*GPropertyLib::NUM_MATSUR ;
    262 
    263     LOG(verbose) << "OBndLib::makeBoundaryOptical obuf "
    264               << obuf->getShapeString()
    265               << " numBnd " << numBnd
    266               << " numBytes " << numBytes
    267               << " nx " << nx
    268               ;
    269 
    270     assert( obuf->getShape(0) == numBnd );
    271 
    272     optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, nx );
    273     memcpy( optical_buffer->map(), obuf->getBytes(), numBytes );
    274     optical_buffer->unmap();
    275 
    276     m_context["optical_buffer"]->setBuffer(optical_buffer);
    277 }






::

     48 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     49 {       
     50     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     51     // >0 outward going photon
     52     // <0 inward going photon
     53     //
     54     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     55     //    it is just 
     56     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     57     //
     58 
     59     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     60 
     61     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     62     // 
     63     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     64     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     65     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     66 
     67     //  consider photons arriving at PMT cathode surface
     68     //  geometry normals are expected to be out of the PMT 
     69     //
     70     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     71 
     72     s.material1 = boundary_lookup( wavelength, m1_line, 0);
     73     s.m1group2  = boundary_lookup( wavelength, m1_line, 1);
     74 
     75     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     76     s.surface   = boundary_lookup( wavelength, su_line, 0);
     77 
     78     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     79 
     80     s.index.x = optical_buffer[m1_line].x ; // m1 index
     81     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     82     s.index.z = optical_buffer[su_line].x ; // su index
     83     s.index.w = identity.w   ;
     84 
     85     s.identity = identity ;
     86 
     87 }


An associated surface is signaled by s.optical.x > 0 for the *su_line*::

    631         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    632         {
    633             command = propagate_at_surface(p, s, rng);
    634             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    635             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    636         }
    637         else
    638         {
    639             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    640             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    641             // tacit CONTINUE
    642         }


And the surface must have non-zero *s.surface.x* to have a chance of *s.flag = SURFACE_DETECT*::


    674 __device__ int
    675 propagate_at_surface(Photon &p, State &s, curandState &rng)
    676 {
    677     float u_surface = curand_uniform(&rng);
    678 #ifdef WITH_ALIGN_DEV
    679     float u_surface_burn = curand_uniform(&rng);
    680 #endif
    681 
    682 #ifdef WITH_ALIGN_DEV_DEBUG
    683     rtPrintf("propagate_at_surface   u_OpBoundary_DiDiAbsorbDetectReflect:%.9g \n", u_surface);
    684     rtPrintf("propagate_at_surface   u_OpBoundary_DoAbsorption:%.9g \n", u_surface_burn);
    685 #endif
    686 
    687     if( u_surface < s.surface.y )   // absorb   
    688     {
    689         s.flag = SURFACE_ABSORB ;
    690         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    691         return BREAK ;
    692     }
    693     else if ( u_surface < s.surface.y + s.surface.x )  // absorb + detect
    694     {
    695         s.flag = SURFACE_DETECT ;
    696         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    697         return BREAK ;
    698     }
    699     else if (u_surface  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    700     {
    701         s.flag = SURFACE_DREFLECT ;
    702         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    703         return CONTINUE;
    704     }
    705     else
    706     {
    707         s.flag = SURFACE_SREFLECT ;
    708         //propagate_at_specular_reflector(p, s, rng );
    709         propagate_at_specular_reflector_geant4_style(p, s, rng );
    710         return CONTINUE;
    711     }
    712 }



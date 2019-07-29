bulletproof_alignment_debug
==============================

Point-by-point random consumption cursor ? seqcon 
---------------------------------------------------

* want a better way to judge real random alignment, storing 
  random floats is profligate, taking 4 bytes for just one, 
  as did with --utaildebug and it is not a comprehensive check  

  * :doc:`ts-box-utaildebug-decouple-maligned-from-deviant`

  * that doc does some u-counting : showing that apart from SC where 
    do not known how many turns of the scattering loop (which each consume 5u) 
    can predict the consumption from the history 


* how about storing point-by-point random cursor index for the consumption ?  

  * thats bulletproof : ie if it matches are as sure as is possible 
    that are really aligned 
  * CPU side CRandomEngine has m_cursor already, need a something similar 
    but per point ? 
  
    * start with m_step_cursors vector which collects in CRandomEngine::postStep
    * actually better to follow the (flag, material) pattern of feeding these
      in point by point from CRecorder level : because thats close to what will 
      be done GPU side 

  * how to do it GPU side ?

    * replace curand_uniform with a wrapper that keeps count 
      in a generate.cu global ? or just manually increament a count 
      for each curand_uniform and pack it into seqcon just like seqhis

* hmm < 256 randoms is the total per photon ? are the consumption
  per step point always less than 16 ?  If so can store point by point 
  consumption index into a "seqcon" for up to 16 points.
  
* hmm need to deal with the kludge nudges G4 side 








Want to visualize where seqcon gets out of step ?
------------------------------------------------------

* ab seqcon comparison can yield an abseqcon array, hmm need some 
  way to add an adhoc photon length abseqcon  


Want to visualize where zeroSteps or other wierdnesses happen, how to implement ?
-----------------------------------------------------------------------------------

* Can I squeeze a debug flag into point records ?

  * `CWriter::writeStepPoint_` has some 16 bits waiting for this

* How to visualize that with the OpenGL shaders, when using --vizg4 
  
  * (zeroSteps and the other wierdnesses are G4 only things of course)

* How to select photons with some zeroStep action in their points ?
  

oglrap/gl/fcolor.h
--------------------

::

     flq[0] and flq[1] are two consequtive step points : as this is only used from geometry shader


     15     switch(ColorParam.x)
     16     {
     17        case 0:
     18               fcolor = vec4(1.0,1.0,1.0,1.0) ; break;
     19        case 1:
     20               fcolor = texture(Colors, (float(flq[0].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     21        case 2:
     22               fcolor = texture(Colors, (float(flq[1].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     23        case 3:
     24               fcolor = texture(Colors, (float(flq[0].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     25        case 4:
     26               fcolor = texture(Colors, (float(flq[1].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     27        case 5:
     28               fcolor = vec4(vec3(polarization[0]), 1.0) ; break;
     29        case 6:
     30               fcolor = vec4(vec3(polarization[1]), 1.0) ; break;
     31     }
     32 


fcolor.h used from geometry shader, so has two points available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

::

     epsilon:gl blyth$ grep fcolor.h */*.glsl 
     altrec/geom.glsl:#incl fcolor.h
     devrec/geom.glsl:#incl fcolor.h
     rec/geom.glsl:#incl fcolor.h



Source of the flq 
-------------------

oglrap/gl/rec/vert.glsl::

     01 #version 410 core
      2 
      3 //  rec/vert.glsl
      4 
      5 layout(location = 0) in vec4  rpos;
      6 layout(location = 1) in vec4  rpol;
      7 layout(location = 2) in ivec4 rflg;
      8 layout(location = 3) in ivec4 rsel;
     09 layout(location = 4) in uvec4 rflq;
     10 
     11 out vec4 polarization ;
     12 out uvec4 flags ;
     13 out uvec4 flq ;
     14 out ivec4 sel  ;
     15 
     16 void main ()
     17 {
     18     sel = rsel ;
     19     polarization = rpol ;
     20     flags = rflg ;
     21     flq = rflq ;
     22 
     23     gl_Position = rpos ;
     24     gl_PointSize = 1.0;
     25 }

::

    epsilon:oglrap blyth$ opticks-f rflq
    ./optickscore/OpticksEvent.cc:    ViewNPY* rflq = new ViewNPY("rflq",m_record_data,0,1,2 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);   
    ./optickscore/OpticksEvent.cc:    m_record_attr->add(rflq);
    ./optixrap/cu/photon.h://  * NumpyEvt::setRecordData sets rflq buffer input as ViewNPY::BYTE starting from offset 2 (ie .z) 


::

    1384 void OpticksEvent::setRecordData(NPY<short>* record_data)
    1385 {
    1386     setBufferControl(record_data);
    1387     m_record_data = record_data  ;
    1388 
    1389     //                                               j k l  sz   type                  norm   iatt   item_from_dim
    1390     ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    1391     ViewNPY* rpol = new ViewNPY("rpol",m_record_data,0,1,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);
    1392 
    1393     ViewNPY* rflg = new ViewNPY("rflg",m_record_data,0,1,2 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);
    1394     // NB l=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  
    1395 
    1396     ViewNPY* rflq = new ViewNPY("rflq",m_record_data,0,1,2 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);
    1397     // NB l=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo
    1398 
    1399     m_record_attr = new MultiViewNPY("record_attr");
    1400 
    1401     m_record_attr->add(rpos);
    1402     m_record_attr->add(rpol);
    1403     m_record_attr->add(rflg);
    1404     m_record_attr->add(rflq);
    1405 }


::

    133 __device__ void rsave( Photon& p, State& s, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
    134 {   
    135     rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
    136                     shortnorm(p.position.x, center_extent.x, center_extent.w),
    137                     shortnorm(p.position.y, center_extent.y, center_extent.w),
    138                     shortnorm(p.position.z, center_extent.z, center_extent.w),
    139                     shortnorm(p.time      , time_domain.x  , time_domain.y  )
    140                     );
    141     
    142     float nwavelength = 255.f*(p.wavelength - boundary_domain.x)/boundary_domain.w ; // 255.f*0.f->1.f 
    143     
    144     qquad qpolw ;    
    145     qpolw.uchar_.x = __float2uint_rn((p.polarization.x+1.f)*127.f) ;  // pol : -1->1  pol+1 : 0->2   (pol+1)*127 : 0->254
    146     qpolw.uchar_.y = __float2uint_rn((p.polarization.y+1.f)*127.f) ;  
    147     qpolw.uchar_.z = __float2uint_rn((p.polarization.z+1.f)*127.f) ;
    148     qpolw.uchar_.w = __float2uint_rn(nwavelength)  ;
    149     
    150     // tightly packed, polarization and wavelength into 4*int8 = 32 bits (1st 2 npy columns) 
    151     hquad polw ;    // union of short4, ushort4
    152     polw.ushort_.x = qpolw.uchar_.x | qpolw.uchar_.y << 8 ;
    153     polw.ushort_.y = qpolw.uchar_.z | qpolw.uchar_.w << 8 ;
    154     
    155 
    156 #ifdef IDENTITY_CHECK
    157     // spread uint32 photon_id across two uint16
    158     unsigned int photon_id = p.flags.u.y ;    
    159     polw.ushort_.z = photon_id & 0xFFFF ;     // least significant 16 bits first     
    160     polw.ushort_.w = photon_id >> 16  ;       // arranging this way allows scrunching to view two uint16 as one uint32 
    161     // OSX intel + CUDA GPUs are little-endian : increasing numeric significance with increasing memory addresses 
    162 #endif
    163      
    164      // boundary int and m1 index uint are known to be within char/uchar ranges 
    165     //  uchar: 0 to 255,   char: -128 to 127 
    166     
    167     qquad qaux ;     // quarter sized quads 
    168     qaux.uchar_.x =  s.index.x ;    // m1  
    169     qaux.uchar_.y =  s.index.y ;    // m2   
    170     qaux.char_.z  =  p.flags.i.x ;  // boundary(range -55:55)   debugging some funny material codes
    171     qaux.uchar_.w = __ffs(s.flag) ; // first set bit __ffs(0) = 0, otherwise 1->32 
    172     
    173     //             lsb_ (flq[0].x)    msb_ (flq[0].y) 
    174     polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    175     
    176     //              lsb_ (flq[0].z)    msb_ (flq[0].w)
    177     polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;
    178     
    179     rbuffer[record_offset+1] = polw.short_ ;
    180 }




::

    168 void CWriter::writeStepPoint_(const G4StepPoint* point, const CPhoton& photon )
    169 {
    170     // write compressed record quads into buffer at location for the m_record_id 
    171 
    172     unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ;
    173     unsigned slot = photon._slot_constrained ;
    174     unsigned flag = photon._flag ;
    175     unsigned material = photon._mat ;
    ...
    228     qquad qaux ;
    229     qaux.uchar_.x = material ;
    230     qaux.uchar_.y = 0 ; // TODO:m2 
    231     qaux.char_.z  = 0 ; // TODO:boundary (G4 equivalent ?)
    232     qaux.uchar_.w = BBit::ffs(flag) ;   // ? duplicates seqhis  

    ^^^^^^^^^^  clearly this is the place to use ^^^^^^^^^^^^^^^^^^^

    233 
    234     hquad polw ;
    235     polw.ushort_.x = polx | poly << 8 ;
    236     polw.ushort_.y = polz | wavl << 8 ;
    237     polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    238     polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;
    239 
    240     //unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 
    241 
    242     m_target_records->setQuad(target_record_id, slot, 0, posx, posy, posz, time_ );
    243     m_target_records->setQuad(target_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );
    244 
    245     // dynamic mode : fills in slots into single photon dynamic_records structure 
    246     // static mode  : fills directly into a large fixed dimension records structure
    247 
    248     // looks like static mode will succeed to scrub the AB and replace with RE 
    249     // just by decrementing m_slot and running again
    250     // but dynamic mode will have an extra record
    251 }




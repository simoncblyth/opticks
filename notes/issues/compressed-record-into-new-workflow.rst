compressed-record-into-new-workflow
=====================================

reviewing OpticksEvent, clearly QEvent needs to replace it in new workflow
-----------------------------------------------------------------------------

* not aiming for compatibility, but QEvent needs to be reminscent of OpticksEvent 
  in order to make updating the numpy analysis machinery to use persisted QEvents easier 


What does QEvent lack compared to OpticksEvent ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* many things that will try living without 
* seqhis+seqmat sequence array (photon level 64-bit uint) 
  
  * THIS IS A MUST HAVE AS ITS THE BASIS FOR HISTORY TABLES 
  * implementing in sseq.h 

what about persisting directory layout ?

::

    2110 void OpticksEvent::savePhotonData()
    2111 {
    2112     NPY<float>* ox = getPhotonData();
    2113     if(ox) ox->save(m_pfx, "ox", m_typ,  m_tag, m_udet);
    2114 }
 
Am inclined for QEvent directory handling to be deferred to the 
user of QEvent with QEvent just concerned with the content of the directory. 



hmm : need to get the domains ce/td/wd onto device for encoding the srec and into python and OpenGL for analysis
-----------------------------------------------------------------------------------------------------------------

* where to carry the domains ? they are only needed for compressed rec so it belongs in NP rec metadata on host. 

  * managing this is qevent.h  

* on device need to keep it somewhere like in qevent ? so QEvent needs to orchestrate the domains. 


using domains from OpenGL shaders : how ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Was expecting uniforms for all the domains, but seems only TimeDomain done that way, see oglrap/gl/rec/geom.glsl 

* trick is to view the data with different types, offsets, iatt etc.. 

::

    1739 void OpticksEvent::setRecordData(NPY<short>* record_data)
    1740 {
    1741     setBufferControl(record_data);
    1742     m_record_data = record_data  ;
    1743     
    1744     //                                               j k l  sz   type                  norm   iatt   item_from_dim
    1745     ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0 ,4,ViewNPY::SHORT          ,true,  false, 2);
    1746     ViewNPY* rpol = new ViewNPY("rpol",m_record_data,0,1,0 ,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);     
    1747 
    1748     ViewNPY* rflg = new ViewNPY("rflg",m_record_data,0,1,2 ,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);     // UNSIGNED_SHORT 16 bit 
    1749     // NB l=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  
    1750 
    1751     ViewNPY* rflq = new ViewNPY("rflq",m_record_data,0,1,2 ,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);     // UNSIGNED_BYTES  8 bit 
    1752     // NB l=2 again : UBYTE view of the same data for access to  m1,m2,boundary,flag
    1753 
    1754     

Expect can replace this old heavy approach (ViewNPY/MultiViewNPY) with just some attribute metadata 
strings associated with the array data. 

* NB attribute type is independant of the array type 




The attribute metadata needs to carry what is needed for attribute setup::

    404 void Rdr::address(ViewNPY* vnpy)
    405 {
    406     const char* name = vnpy->getName();
    407     GLint location = m_shader->attribute(name, false);
    ...
    415     GLenum type = GL_FLOAT  ;              //  of each component in the array
    416     switch(vnpy->getType())
    417     {   
    418         case ViewNPY::BYTE:                         type = GL_BYTE           ; break ;
    419         case ViewNPY::UNSIGNED_BYTE:                type = GL_UNSIGNED_BYTE  ; break ;
    420         case ViewNPY::SHORT:                        type = GL_SHORT          ; break ;
    421         case ViewNPY::UNSIGNED_SHORT:               type = GL_UNSIGNED_SHORT ; break ;
    422         case ViewNPY::INT:                          type = GL_INT            ; break ;
    423         case ViewNPY::UNSIGNED_INT:                 type = GL_UNSIGNED_INT   ; break ;
    424         case ViewNPY::HALF_FLOAT:                   type = GL_HALF_FLOAT     ; break ;
    425         case ViewNPY::FLOAT:                        type = GL_FLOAT          ; break ;     
    426         case ViewNPY::DOUBLE:                       type = GL_DOUBLE         ; break ;     
    427         case ViewNPY::FIXED:                        type = GL_FIXED                        ; break ;
    428         case ViewNPY::INT_2_10_10_10_REV:           type = GL_INT_2_10_10_10_REV           ; break ; 
    429         case ViewNPY::UNSIGNED_INT_2_10_10_10_REV:  type = GL_UNSIGNED_INT_2_10_10_10_REV  ; break ; 
    430         //case ViewNPY::UNSIGNED_INT_10F_11F_11F_REV: type = GL_UNSIGNED_INT_10F_11F_11D_REV ; break ; 
    431         default: assert(0)                                                                 ; break ;
    432     }
    ...
    461     if( vnpy->getIatt() )
    462     {
    463         glVertexAttribIPointer(index, size, type, stride, offset);
    464     }
    465     else
    466     {
    467         glVertexAttribPointer(index, size, type, norm, stride, offset);
    468     }


* in new workflow the natural place to parse the attribute metadata is SGLFW 

* https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glVertexAttribPointer.xhtml

::

    void glVertexAttribPointer( 	
        GLuint index,
        GLint size,
        GLenum type,
        GLboolean normalized,
        GLsizei stride,
        const void * pointer);

::
    // size,type,normalized,stride,offset,iatt 

    att_vpos:4,GL_SHORT,1,16,0,0  



new way of managing domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* do not like depending on OpticksCore for such a basic thing, domains belongs in QEvent 
* needed both on GPU/CPU and are constants so should live in qevent 


old way of managing domains on device was simply OptiX context globals grabbed from Opticks::getSpaceDomain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:optixrap blyth$ grep boundary_domain *.*
    OBndLib.cc:    m_context["boundary_domain"]->setFloat(dom.x, dom.y, dom.z, dom.w); 
    OBndLib.cc:    m_context["boundary_domain_reciprocal"]->setFloat(rdom.x, rdom.y, rdom.z, rdom.w); 
    OBndLib.cc:        << "boundary_domain_reciprocal "

    epsilon:optixrap blyth$ grep time_domain *.*
    OPropagator.cc:    m_context["time_domain"]->setFloat(   make_float4( td.x, td.y, td.z, td.w ));

    epsilon:optixrap blyth$ grep center_extent *.*
    OPropagator.cc:    m_context["center_extent"]->setFloat( make_float4( ce.x, ce.y, ce.z, ce.w ));


oxrap/OPropagator.cc::

    104 void OPropagator::initParameters()
    105 {
    ...
    135     const glm::vec4& ce = m_ok->getSpaceDomain();
    136     const glm::vec4& td = m_ok->getTimeDomain();
    137 
    138     m_context["center_extent"]->setFloat( make_float4( ce.x, ce.y, ce.z, ce.w ));
    139     m_context["time_domain"]->setFloat(   make_float4( td.x, td.y, td.z, td.w ));
    140 }
    141 



ocu/boundary_lookup.h which gets included into generate.cu::

    038 #include "GPropertyLib.hh"
     39 
     40 rtTextureSampler<float4, 2>  boundary_texture ;
     41 rtDeclareVariable(float4, boundary_domain, , );
     42 rtDeclareVariable(float4, boundary_domain_reciprocal, , );
     43 rtDeclareVariable(uint4,  boundary_bounds, , );
     44 rtDeclareVariable(uint4,  boundary_texture_dim, , );

ocu/generate.cu::

    131 rtDeclareVariable(float4,        center_extent, , );
    132 rtDeclareVariable(float4,        time_domain  , , );
    133 rtDeclareVariable(uint4,         debug_control , , );
    134 rtDeclareVariable(float,         propagate_epsilon, , );


    192 #define RSAVE(seqhis, seqmat, p, s, slot, slot_offset)  \
    193 {    \
    194     unsigned int shift = slot*4 ; \
    195     unsigned long long his = __ffs((s).flag) & 0xF ; \
    196     unsigned long long mat = (s).index.x < 0xF ? (s).index.x : 0xF ; \
    197     seqhis |= his << shift ; \
    198     seqmat |= mat << shift ; \
    199     rsave((p), (s).flag, (s).index, _record_buffer, slot_offset*RNUMQUAD , center_extent, time_domain, boundary_domain );  \
    200 }   \
    201 

ocu/photon.h::

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


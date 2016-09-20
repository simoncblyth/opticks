OptiX CUDA Interop With 3080
=================================

Attempting to use UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY OR BUFFER_COPY_ON_DIRTY 
with the seed buffer in 3080 yields a hard CUDA crash on 2nd event launch, even 
with dumpseed.

Success in 3080 with okop-/tests/dirtyBufferTest made me get rid 
of UPLOAD_WITH_CUDA,BUFFER_COPY_ON_DIRTY.

Initially however the stuck at zero issue remained


FIX
-----

The below commit succeeds to fix issue.

* https://bitbucket.org/simoncblyth/opticks/commits/7387baedddb53897a4d710b0a59b78157efec6de


Initially it looked like the use of a prelaunch between 
gensteps upload and seeding was the reason for success, but subsequently 
it is operational without that... so there is some other cause.

Examining the commit the other significant change is to not try 
to download the OPTIX_INPUT_ONLY seed buffer. And remove the BUFFER_COPY_ON_DIRTY
from seed buffer. 
Potentially those messed things up ?


Reference
-----------

::

   
   open /Developer/OptiX_380/doc/OptiX_Programming_Guide_3.8.0.pdf


Overview of seeding interop
-------------------------------

Seeding OpSeeder::seedPhotonsFromGenstepsImp is done via Thrust, dumping 
seed buffer shows is as expected on all events, but it is only seen
by OptiX on the first::

   254 #ifdef WITH_SEED_BUFFER
   255     tox.dump<unsigned int>("OpSeeder::seedPhotonsFromGenstepsImp tox.dump", 1*1, 0, num_photons ); // stride, begin, end 
   256 #endif

Try skipping the seed buffer zero ahead, makes no difference.

So problem is Thrust to OptiX handover.


::

    052 void OpSeeder::seedPhotonsFromGensteps()
     53 {
     54     LOG(info)<<"OpSeeder::seedPhotonsFromGensteps" ;
     55     if( m_ocontext->isInterop() )
     56     {
     57 #ifdef WITH_SEED_BUFFER
     58         seedComputeSeedsFromInteropGensteps();
     59 #else
     60         seedPhotonsFromGenstepsViaOpenGL();
     61 #endif
     62     }   
     63     else if ( m_ocontext->isCompute() )
     64     {   
     65         seedPhotonsFromGenstepsViaOptiX();
     66     }   
     67     if(m_hub->hasOpt("onlyseed")) exit(EXIT_SUCCESS);
     68 }




    022 CBufSlice OBufBase::slice( unsigned int stride, unsigned int begin, unsigned int end )
     23 {     
     24    return CBufSlice( getDevicePtr(), getSize(), getNumBytes(), stride, begin, end == 0u ? getNumAtoms() : end);
     25 }     
     26       
     27 CBufSpec OBufBase::bufspec()
     28 {  
     29    return CBufSpec( getDevicePtr(), getSize(), getNumBytes()) ;
     30 }     

    /// CBufSpec is just a holder for dev_ptr together with size and numBytes


    175 void* OBufBase::getDevicePtr()
    176 {
    177     //printf("OBufBase::getDevicePtr %s \n", ( m_name ? m_name : "-") ) ;
    178     //return (void*) m_buffer->getDevicePointer(m_device); 
    179 
    180     CUdeviceptr cu_ptr = (CUdeviceptr)m_buffer->getDevicePointer(m_device) ;
    181     return (void*)cu_ptr ;
    182 }





    141 void OpSeeder::seedPhotonsFromGenstepsViaOptiX()
    142 {
    143     OK_PROFILE("_OpSeeder::seedPhotonsFromGenstepsViaOptiX");
    144 
    145     OBuf* genstep = m_oevt->getGenstepBuf() ;
    146     CBufSpec s_gs = genstep->bufspec();     /// just a holder for dev_ptr, size, num_bytes
    147 
    148 #ifdef WITH_SEED_BUFFER
    149     LOG(info) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : SEEDING TO SEED BUF  " ;
    150     OBuf* seed = m_oevt->getSeedBuf() ;
    151     CBufSpec s_se = seed->bufspec();  //  optix::Buffer::getDevicePointer happens here 
    152     seedPhotonsFromGenstepsImp(s_gs, s_se);
    153     //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");
    154     //s_se.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_se");
    155 #else
    156     LOG(info) << "OpSeeder::seedPhotonsFromGenstepsViaOptiX : seeding to photon buf  " ;
    157     OBuf* photon = m_oevt->getPhotonBuf() ;
    158     CBufSpec s_ox = photon->bufspec();
    159     seedPhotonsFromGenstepsImp(s_gs, s_ox);
    160 #endif
    161 
    162     //genstep->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)genstep");
    163     //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_gs");
    164 
    165     //photon->Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (OBuf)photon ");
    166     //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsViaOptiX (CBufSpec)s_ox");
    167 
    168 
    169 
    170     TIMER("seedPhotonsFromGenstepsViaOptiX");
    171     OK_PROFILE("OpSeeder::seedPhotonsFromGenstepsViaOptiX");
    172 
    173 }





    208 void OpSeeder::seedPhotonsFromGenstepsImp(const CBufSpec& s_gs, const CBufSpec& s_ox)
    209 {
    210     //s_gs.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_gs");
    211     //s_ox.Summary("OpSeeder::seedPhotonsFromGenstepsImp (CBufSpec)s_ox");
    212 
    213     TBuf tgs("tgs", s_gs );
    214     TBuf tox("tox", s_ox );


::

     19 TBuf::TBuf(const char* name, CBufSpec spec ) :
     20         m_name(strdup(name)),
     21         m_spec(spec)
     22 {     
     23 }
     24       
     25 CBufSlice TBuf::slice( unsigned int stride, unsigned int begin, unsigned int end ) const
     26 {     
     27     if(end == 0u) end = m_spec.size ;
     28     return CBufSlice(m_spec.dev_ptr, m_spec.size, m_spec.num_bytes, stride, begin, end);
     29 }     
     ..
     36 void* TBuf::getDevicePtr() const 
     37 {
     38     return m_spec.dev_ptr ; 
     39 }






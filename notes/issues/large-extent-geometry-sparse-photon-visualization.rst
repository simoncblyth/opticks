large-extent-geometry-sparse-photon-visualization
=====================================================

context
------------

* :doc:`tboolean-proxy-scan`

Need to debug issues with some large extent solids, and viz has
problems with large extent.



command shortcuts
--------------------

::

    lv(){ echo 10 ; }
    # default geometry LV index to test 

    ts(){  PROXYLV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  PROXYLV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation



rejigged shortcuts moving the above options within tboolean-lv
------------------------------------------------------------------

::

    [blyth@localhost ana]$ t opticks-tboolean-shortcuts
    opticks-tboolean-shortcuts is a function
    opticks-tboolean-shortcuts () 
    { 
        : default geometry LV index or tboolean-geomname eg "box" "sphere" etc..;
        function lv () 
        { 
            echo 21
        };
        : **simulate** : aligned bi-simulation creating OK+G4 events;
        function ts () 
        { 
            LV=${1:-$(lv)} tboolean.sh $*
        };
        : **visualize** : load events and visualize the propagation;
        function tv () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load $*
        };
        : **visualize** the geant4 propagation;
        function tv4 () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load --vizg4 $*
        };
        : **analyse** : load events and analyse the propagation;
        function ta () 
        { 
            LV=${1:-$(lv)} tboolean.sh --ip
        }
    }



ISSUE 2 : times(ns) of propagation milestones (eg BT) shown on animation slider vary as change AnimTimeMax
----------------------------------------------------------------------------------------------------------------

* hmm some scaling missing ?

::

   TMAX=-1 ts 10    # auto timeMax, base on extent 

   TMAX=-1 ta 10    # auto animTimeMax based on extent
                    # hmm probably makes more sense to use value from fdom ?  


ta 10::

    In [2]: ab.aselhis = "TO BT BT SA"

    In [3]: a.rpost().shape
    Out[3]: (7610, 4, 4)

    In [4]: a.rpost()
    Out[4]: 
    A()sliced
    A([[[  1806.2326,  -5543.9474, -71998.8026,      0.    ],
        [  1806.2326,  -5543.9474,  -2500.5993,    231.8218],
        [  1806.2326,  -5543.9474,   2500.5993,    262.1235],
        [  1806.2326,  -5543.9474,  72001.    ,    493.9453]],

       [[ -4269.4767,    -46.1446, -71998.8026,      0.    ],
        [ -4269.4767,    -46.1446,  -2500.5993,    231.8218],
        [ -4269.4767,    -46.1446,   2500.5993,    262.1235],
        [ -4269.4767,    -46.1446,  72001.    ,    493.9453]],



    In [5]: ab.fdom
    Out[5]: 
    A(torch,1,tboolean-proxy-10)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[    0.  ,     0.  ,     0.  , 72001.  ]],

       [[    0.  ,   720.01,   720.01,     0.  ]],

       [[   60.  ,   820.  ,    20.  ,   760.  ]]], dtype=float32)




Viz with auto time domain::

    TMAX=-1 tv 10


* its a bit difficult to select times with the slider precisely, because of the animation steps and great sensitivity when zoomed in 
* but it looks like at 231.20/261.42 ns on slider the photons are just before the lower/upper boundaries 
  which is very close to rpost() values 

Now with changing animTimeMax::

    TMAX=250 tv 10 

* at 80.28/90.77 in slider are just before boundaries
* what about a non-zero animTimeMin ?

Hmm, looks like need to scale by animTimeMax/timeMax for the slider numbers to be correct::

    In [1]: 250./720.
    Out[1]: 0.3472222222222222

    In [2]: 231.2*0.3472222222
    Out[2]: 80.27777777264

    In [3]: 261.42*0.347222222
    Out[3]: 90.77083327524001

oglrap/gl/rec/geom.glsl::

     31 out vec4 fcolor ;
     32 
     33 void main ()
     34 {
     35     uint seqhis = sel[0].x ;
     36     uint seqmat = sel[0].y ;
     37     if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
     38     if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;
     39 
     40     uint photon_id = gl_PrimitiveIDIn/MAXREC ;
     41     if( PickPhoton.x > 0 && PickPhoton.y > 0 && PickPhoton.x != photon_id )  return ;
     42 
     43 
     44     vec4 p0 = gl_in[0].gl_Position  ;
     45     vec4 p1 = gl_in[1].gl_Position  ;
     46     float tc = Param.w / TimeDomain.y ;
     47 
     48     uint valid  = (uint(p0.w >= 0.)  << 0) + (uint(p1.w >= 0.) << 1) + (uint(p1.w > p0.w) << 2) ;
     49     uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;
     50     uint vselect = valid & select ;
     51 
     52 #incl fcolor.h
     53 
     54     if(vselect == 0x7) // both valid and straddling tc
     55     {
     56         vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
     57         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     58 
     59         if(NrmParam.z == 1)
     60         {
     61             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     62             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     63         }
     64 
     65 
     66         EmitVertex();
     67         EndPrimitive();
     68     }
     69     else if( valid == 0x7 && select == 0x5 )     // both valid and prior to tc
     70     {
     71         vec3 pt = vec3(p1) ;
     72         gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ;
     73 
     74         if(NrmParam.z == 1)
     75         {
     76             float depth = ((gl_Position.z / gl_Position.w) + 1.0) * 0.5;
     77             if(depth < ScanParam.x || depth > ScanParam.y ) return ;
     78         }
     79 



p0,p1 
    rpos domain compressed positions and times and will be in range -1.f:1.f
    using the position and time domains active at simulation 

Param.w
    uniform propagation time coming from the Animator (or slider) which 
    is in range m_domain_time.x(timemin), m_domain_time.z(animTimeMax)
 
TimeDomain.y
    from Composition::getTimeDomain which is set at OpticksViz::uploadGeometry 
    using Opticks::getTimeDomain 
    

Keeping animTimeMax the same as timeMax avoids the problem.



::

    317 void OpticksViz::uploadGeometry()
    318 {
    319     LOG(LEVEL) << "[ hub " << m_hub->desc() ;
    320 
    321     NPY<unsigned char>* colors = m_hub->getColorBuffer();
    322 
    323     m_scene->uploadColorBuffer( colors );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"
    324 
    325     LOG(info) << m_ok->description();
    326 
    327     m_composition->setTimeDomain(        m_ok->getTimeDomain() );
    328     m_composition->setDomainCenterExtent(m_ok->getSpaceDomain());
    329 
    330     m_scene->setGeometry(m_ggb->getGeoLib());
    331 
    332     m_scene->uploadGeometry();
    333 
    334 
    335     m_hub->setupCompositionTargetting();
    336 
    337     LOG(LEVEL) << "]" ;
    338 }



::

    1998 void Opticks::setupTimeDomain(float extent)
    1999 {
    2000     float timemax = m_cfg->getTimeMax();  // ns
    2001     float animtimemax = m_cfg->getAnimTimeMax() ;
    2002 
    2003     float speed_of_light = 300.f ;        // mm/ns 
    2004     float rule_of_thumb_timemax = 3.f*extent/speed_of_light ;
    2005 
    2006     float u_timemin = 0.f ;  // ns
    2007     float u_timemax = timemax < 0.f ? rule_of_thumb_timemax : timemax ;
    2008     float u_animtimemax = animtimemax < 0.f ? u_timemax : animtimemax ;
    2009 
    2010     LOG(info)
    2011         << " cfg.getTimeMax [--timemax] " << timemax
    2012         << " cfg.getAnimTimeMax [--animtimemax] " << animtimemax
    2013         << " speed_of_light (mm/ns) " << speed_of_light
    2014         << " extent (mm) " << extent
    2015         << " rule_of_thumb_timemax (ns) " << rule_of_thumb_timemax
    2016         << " u_timemax " << u_timemax
    2017         << " u_animtimemax " << u_animtimemax
    2018         ;
    2019 
    2020     m_time_domain.x = u_timemin ;
    2021     m_time_domain.y = u_timemax ;
    2022     m_time_domain.z = u_animtimemax ;
    2023     m_time_domain.w = 0.f  ;
    2024 }
    2025 





Hmm using m_domain_time.z  AnimTimeMax::

     776 void Composition::initAnimator()
     777 {
     778     float* target = glm::value_ptr(m_param) + 3 ;   // offset to ".w" 
     779 
     780     m_animator = new Animator(target, m_animator_period, m_domain_time.x, m_domain_time.z );
     781     m_animator->setModeRestrict(Animator::FAST);
     782     m_animator->Summary("Composition::gui setup Animation");
     783 }









oglrap/Rdr.cc::

    477 void Rdr::update_uniforms()
    478 {
    479 
    480     if(m_composition)
    481     {
    482         // m_composition->update() ; moved up to Scene::render
    ...
    501         glm::vec4 par = m_composition->getParam();
    502         glUniform4f(m_param_location, par.x, par.y, par.z, par.w  );
    ...
    518         glm::vec4 td = m_composition->getTimeDomain();
    519         glUniform4f(m_timedomain_location, td.x, td.y, td.z, td.w  );



::

    [blyth@localhost optickscore]$ opticks-f setTimeDomain
    ./npy/RecordsNPY.cpp:void RecordsNPY::setTimeDomain(glm::vec4& td)
    ./npy/RecordsNPY.cpp:    setTimeDomain(td);    
    ./npy/RecordsNPY.hpp:       void setTimeDomain(glm::vec4& td);
    ./oglrap/OpticksViz.cc:    m_composition->setTimeDomain(        m_ok->getTimeDomain() );
    ./optickscore/Opticks.cc:    evt->setTimeDomain(getTimeDomain());
    ./optickscore/OpticksEvent.cc:void OpticksEvent::setTimeDomain(const glm::vec4& time_domain) {             m_domain->setTimeDomain(time_domain)  ; } 
    ./optickscore/OpticksDomain.cc:void OpticksDomain::setTimeDomain(const glm::vec4& time_domain)
    ./optickscore/OpticksDomain.hh:       void setTimeDomain(const glm::vec4& time_domain);
    ./optickscore/Composition.cc:void Composition::setTimeDomain(const glm::vec4& td)
    ./optickscore/Composition.hh:      void setTimeDomain(const glm::vec4& td);
    ./optickscore/OpticksEvent.hh:       void setTimeDomain(const glm::vec4& time_domain);




ISSUE  1 : visualization of photon propagations within large extent volumes is broken
----------------------------------------------------------------------------------------

* get a few 10s of photons only, and they do not go far 


::

    TMAX=500 tv 10
    TMAX=1000 tv 10
    TMAX=2000 tv 10

    TMAX=2000 tv 17


     571 tboolean--(){
     572 
     573     tboolean-
     574 
     575     local msg="=== $FUNCNAME :"
     576     local cmdline=$*
     577 
     578     local stack=2180  # default
     579 
     580     local testname=$(tboolean-testname)
     581     local testconfig=$(tboolean-testconfig)
     582     local torchconfig=$(tboolean-torchconfig)
     583     local tmax=${TMAX:-20}
     584 
     585     tboolean-info
     586 
     587     # $testname--   
     588     #     this assumes testname matches bash function name
     589     #     which is not the case fot tboolean-proxy 
     590 
     591     o.sh  \
     592             $cmdline \
     593             --envkey \
     594             --rendermode +global,+axis \
     595             --animtimemax $tmax \
     596             --timemax $tmax \
     597             --geocenter \
     598             --stack $stack \
     599             --eye 1,0,0 \
     600             --test \
     601             --testconfig "$testconfig" \
     602             --torch \
     603             --torchconfig "$torchconfig" \
     604             --torchdbg \
     605             --tag $(tboolean-tag) \
     606             --pfx $testname \
     607             --cat $testname \
     608             --anakey tboolean \
     609             --args \
     610             --save
     611 



Rule of thumb for picking time domain based on extent ?
------------------------------------------------------------

ta 10::

    In [4]: a.oxa[:,0,3].max()
    Out[4]: 
    A(1216.575, dtype=float32)

    In [5]: a.oxa[:,0,3].min()
    Out[5]: 
    A(1.8343, dtype=float32)

    In [6]: a.fdom
    Out[6]: 
    A(torch,1,tboolean-proxy-10)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[    0.,     0.,     0., 72001.]],

       [[    0.,    20.,    20.,     0.]],
        

       [[   60.,   820.,    20.,   760.]]], dtype=float32)

    In [7]: 72001./300.      ## 300mm/ns
    Out[7]: 240.00333333333333

    In [8]: 2*72001./300.    ##    
    Out[8]: 480.00666666666666


::

    [blyth@localhost ana]$ OKTest -h | grep max
                                     for example      --anakeyargs "--c2max_0.5"   
      --rngmax arg                   Maximum number of photons that can be 
      -b [ --bouncemax ] arg         Maximum number of boundary bounces, 0:prevents
      -r [ --recordmax ] arg         Maximum number of photon step records per 
      --timemax arg                  Maximum time in nanoseconds. Default 
      --animtimemax arg              Maximum animation time in nanoseconds. Default


Increasing TMAX doesnt change the sparse photon viz with large extent solids::

    TMAX=500 tv 10
    TMAX=2000 tv 10


Thats because its insufficient to just load the old event with changed TMAX that only
effects the animation speed not the actual propagation records.






hmm -ve times ?
~~~~~~~~~~~~~~~~~~~

::

    In [15]: a.sel = "TO BT BR BT SC SA"

    In [17]: a.rpost()
    Out[17]: 
    A()sliced
    A([[[  5614.263 ,  -5680.1839, -71998.8026,      0.    ],
        [  5614.263 ,  -5680.1839,  -2500.5993,    231.8218],
        [  5614.263 ,  -5680.1839,   2500.5993,    262.1308],
        [  5614.263 ,  -5680.1839,  -2500.5993,    292.4251],
        [  5614.263 ,  -5680.1839, -29752.2977,    383.3227],
        [ 72001.    ,   2478.6257, -46759.8889,   -480.0213]],




what does "--timemax" "--animtimemax" do exactly ? 
-------------------------------------------------------

* "--timemax" defines the time_domain together with a default zero timemin
  which is used by domain compression of the step point record times

* HENCE : have to simulate again, as "--timemax" is not just a visualization thing 

* insufficient time domain borks the photon step records saved in the event

* subsequent loading the events and visualizing with different time domain
  will do nothing, as not writing events. And the visualization will be broken because 
  times will rapidly saturate the available bits.  

* YEP confirmed, the below succeeds to create useful visualizations with large extent, 
  with scattering it makes for good fireworks  


::

    TMAX=500 ts 10
    TMAX=500 tv 10
 

ta 10::

    In [1]: a.fdom
    Out[1]: 
    A(torch,1,tboolean-proxy-10)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[    0.,     0.,     0., 72001.]],

       [[    0.,   500.,   500.,     0.]],
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  timemax  animtimemax

       [[   60.,   820.,    20.,   760.]]], dtype=float32)




How to automate setting --timemax ?
--------------------------------------

* use a negative time to indicate want to automate it, thence try rule of thumb 2.*extent(mm)/300(mm/ns) 


::

    1980 /**
    1981 Opticks::setupTimeDomain
    1982 -------------------------
    1983 
    1984 When configured values of "--timemax" and "--animtimemax" are 
    1985 negative a rule of thumb is used to setup a timedomain 
    1986 suitable for the extent of space domain.
    1987 
    1988 **/
    1989 
    1990 void Opticks::setupTimeDomain(float extent)
    1991 {
    1992     float timemax = m_cfg->getTimeMax();  // ns
    1993     float animtimemax = m_cfg->getAnimTimeMax() ;
    1994     
    1995     float speed_of_light = 300.f ;        // mm/ns 
    1996     float rule_of_thumb_timemax = 2.f*extent/speed_of_light ;
    1997     
    1998     float u_timemin = 0.f ;  // ns
    1999     float u_timemax = timemax < 0.f ? rule_of_thumb_timemax : timemax ;
    2000     float u_animtimemax = animtimemax < 0.f ? u_timemax : animtimemax ;
    2001     
    2002     LOG(info)
    2003         << " cfg.getTimeMax [--timemax] " << timemax
    2004         << " cfg.getAnimTimeMax [--animtimemax] " << animtimemax
    2005         << " speed_of_light (mm/ns) " << speed_of_light
    2006         << " extent (mm) " << extent  
    2007         << " rule_of_thumb_timemax (ns) " << rule_of_thumb_timemax
    2008         << " u_timemax " << u_timemax
    2009         << " u_animtimemax " << u_animtimemax
    2010         ;  
    2011 
    2012     m_time_domain.x = u_timemin ;
    2013     m_time_domain.y = u_timemax ;
    2014     m_time_domain.z = u_animtimemax ;
    2015     m_time_domain.w = 0.f  ;
    2016 }   


::

    TMAX=-1 ts 10 
   



Tracing getTimeMax getAnimTimeMax
-------------------------------------

OpticksCfg.cc::

    . 88        m_recordmax(10),
      89        m_timemax(200),
      90        m_animtimemax(50),
      91        m_animator_period(200),


    [blyth@localhost optickscore]$ opticks-f getTimeMax
    ./optickscore/Opticks.hh:       float getTimeMax();
    ./optickscore/Opticks.cc:float Opticks::getTimeMax()
    ./optickscore/Opticks.cc:   m_time_domain.y = m_cfg->getTimeMax() ;
    ./optickscore/Opticks.cc:    dd->add("MAXTIME",m_cfg->getTimeMax());    
    ./optickscore/Composition.cc:    //  m_domain_time.y  end      getTimeMax()       (200ns ) 
    ./optickscore/OpticksCfg.cc:float OpticksCfg<Listener>::getTimeMax() const   // --timemax
    ./optickscore/OpticksCfg.hh:     float        getTimeMax() const ; 

::

    [blyth@localhost optickscore]$ opticks-f MAXTIME
    ./optickscore/Opticks.cc:    dd->add("MAXTIME",m_cfg->getTimeMax());

    2272 BDynamicDefine* Opticks::makeDynamicDefine()
    2273 {
    2274     BDynamicDefine* dd = new BDynamicDefine();   // configuration used in oglrap- shaders
    2275     dd->add("MAXREC",m_cfg->getRecordMax());
    2276     dd->add("MAXTIME",m_cfg->getTimeMax());
    2277     dd->add("PNUMQUAD", 4);  // quads per photon
    2278     dd->add("RNUMQUAD", 2);  // quads per record 
    2279     dd->add("MATERIAL_COLOR_OFFSET", (unsigned int)OpticksColors::MATERIAL_COLOR_OFFSET );
    2280     dd->add("FLAG_COLOR_OFFSET", (unsigned int)OpticksColors::FLAG_COLOR_OFFSET );
    2281     dd->add("PSYCHEDELIC_COLOR_OFFSET", (unsigned int)OpticksColors::PSYCHEDELIC_COLOR_OFFSET );
    2282     dd->add("SPECTRAL_COLOR_OFFSET", (unsigned int)OpticksColors::SPECTRAL_COLOR_OFFSET );
    2283 
    2284     return dd ;
    2285 }


    [blyth@localhost opticks]$ opticks-f getAnimTimeMax
    ./optickscore/Opticks.hh:       float getAnimTimeMax() const ; // --animtimemax
    ./optickscore/Opticks.cc:   m_time_domain.z = m_cfg->getAnimTimeMax() ;
    ./optickscore/Opticks.cc:float Opticks::getAnimTimeMax() const 
    ./optickscore/Composition.cc:    //  m_domain_time.z           getAnimTimeMax()   (previously 0.25*TimeMax as all fun in first 50ns)
    ./optickscore/OpticksCfg.cc:float OpticksCfg<Listener>::getAnimTimeMax() const   // --animtimemax
    ./optickscore/OpticksCfg.hh:     float        getAnimTimeMax() const ;  
    [blyth@localhost opticks]$ 



::

    2000 /**
    2001 Opticks::postgeometry
    2002 ------------------------
    2003 
    2004 Invoked by Opticks::setSpaceDomain
    2005 
    2006 **/
    2007 
    2008 void Opticks::postgeometry()
    2009 {
    2010     configureDomains();
    2011 
    2012     m_profile->setDir(getEventFold());
    2013 }
    2014 
    2015 
    2016 void Opticks::configureDomains()
    2017 {
    2018    // this is triggered by setSpaceDomain which is 
    2019    // invoked when geometry is loaded 
    2020    m_domains_configured = true ;
    2021 
    2022    m_time_domain.x = 0.f  ;
    2023    m_time_domain.y = m_cfg->getTimeMax() ;
    2024    m_time_domain.z = m_cfg->getAnimTimeMax() ;
    2025    m_time_domain.w = 0.f  ;
    2026     
    2027    m_wavelength_domain = getDefaultDomainSpec() ;
    2028 
    2029    int e_rng_max = SSys::getenvint("CUDAWRAP_RNG_MAX",-1);
    2030 
    2031    int x_rng_max = getRngMax() ;
    2032    
    2033    if(e_rng_max != x_rng_max)
    2034        LOG(verbose) << "Opticks::configureDomains"
    2035                   << " CUDAWRAP_RNG_MAX " << e_rng_max
    2036                   << " x_rng_max " << x_rng_max
    2037                   ;
    2038    
    2039    //assert(e_rng_max == x_rng_max && "Configured RngMax must match envvar CUDAWRAP_RNG_MAX and corresponding files, see cudawrap- ");    
    2040 }
    2041    
    2042 float Opticks::getTimeMin() const
    2043 {
    2044     return m_time_domain.x ;
    2045 }   
    2046 float Opticks::getTimeMax() const
    2047 {
    2048     return m_time_domain.y ;
    2049 }   
    2050 float Opticks::getAnimTimeMax() const
    2051 {
    2052     return m_time_domain.z ;
    2053 }   
    2054 



     36 void OpticksAim::registerGeometry(GMergedMesh* mm0)
     37 {
     38     m_mesh0 = mm0 ;
     39 
     40     glm::vec4 ce0 = getCenterExtent();
     41     m_ok->setSpaceDomain( ce0 );
     42 
     43     LOG(m_dbgaim ? fatal : LEVEL)
     44           << " setting SpaceDomain : "
     45           << " ce0 " << gformat(ce0)
     46           ;
     47 }

::

    129 __device__ void rsave( Photon& p, State& s, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
    130 {
    131     rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
    132                     shortnorm(p.position.x, center_extent.x, center_extent.w),
    133                     shortnorm(p.position.y, center_extent.y, center_extent.w),
    134                     shortnorm(p.position.z, center_extent.z, center_extent.w),
    135                     shortnorm(p.time      , time_domain.x  , time_domain.y  )
    136                     );


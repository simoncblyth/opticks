tboolean_box_fail_tag_0_not_allowed
======================================

* tag 0 NOT ALLOWED issue : have not seen this in a very long time : why now ? smth about the tds JUNO geometry ?
* hmm looks to be due to the envvar "TAG" being defined  


CONFIRMED : Simple Fix
-----------------------

::

    epsilon:opticks blyth$ git diff integration/tests/tboolean.bash
    diff --git a/integration/tests/tboolean.bash b/integration/tests/tboolean.bash
    index 3564d4204..833b2e8e0 100644
    --- a/integration/tests/tboolean.bash
    +++ b/integration/tests/tboolean.bash
    @@ -520,7 +520,8 @@ tboolean-cd(){  cd $(tboolean-dir); }
     
     join(){ local IFS="$1"; shift; echo "$*"; }
     
    -tboolean-tag(){  echo ${TAG:-1} ; }
    +#tboolean-tag(){  echo ${TAG:-1} ; }
    +tboolean-tag(){  echo ${OPTICKS_EVENT_TAG:-1} ; }
     tboolean-det(){  echo boolean ; }
     tboolean-src(){  echo torch ; }
     tboolean-args(){ echo  --det $(tboolean-det) --src $(tboolean-src) ; }
    epsilon:opticks blyth$ 



Issue onserved with tds JUNO geometry
---------------------------------------

Reproduce with::

   cd ~/opticks/integration/tests
   ./tboolean_box.sh 

    ...
    2021-04-10 23:18:51.735 INFO  [40221] [OGeo::convert@301] [ nmm 10
    2021-04-10 23:18:53.047 INFO  [40221] [OGeo::convert@314] ] nmm 10
    2021-04-10 23:18:53.111 ERROR [40221] [cuRANDWrapper::setItems@154] CAUTION : are resizing the launch sequence 
    2021-04-10 23:18:53.978 FATAL [40221] [ORng::setSkipAhead@160]  skip as as WITH_SKIPAHEAD not enabled 
    2021-04-10 23:18:54.042 FATAL [40221] [OpticksEventSpec::getOffsetTag@90]  iszero itag  pfx tboolean-box typ torch tag O itag 0 det tboolean-box cat tboolean-box eng NO
    OKG4Test: /home/blyth/opticks/optickscore/OpticksEventSpec.cc:96: const char* OpticksEventSpec::getOffsetTag(unsigned int) const: Assertion `!iszero && "--tag 0 NOT ALLOWED : AS USING G4 NEGATED CONVENTION "' failed.

    (gdb) bt
    #3  0x00007fffe5834252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffecc622b1 in OpticksEventSpec::getOffsetTag (this=0x70c0e0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:96
    #5  0x00007fffecc62349 in OpticksEventSpec::clone (this=0x70c0e0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:106
    #6  0x00007fffecc69c6d in OpticksEvent::Make (spec=0x70c0e0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEvent.cc:125
    #7  0x00007fffecc96025 in Opticks::makeEvent (this=0x6d3ed0, ok=true, tagoffset=0) at /home/blyth/opticks/optickscore/Opticks.cc:3299
    #8  0x00007fffecc7ba1d in OpticksRun::createEvent (this=0x6f4ee0, tagoffset=0, cfg4evt=true) at /home/blyth/opticks/optickscore/OpticksRun.cc:111
    #9  0x00007fffecc7b943 in OpticksRun::createEvent (this=0x6f4ee0, gensteps=0x8b2b800, cfg4evt=true) at /home/blyth/opticks/optickscore/OpticksRun.cc:96
    #10 0x00007fffecc8e8e9 in Opticks::createEvent (this=0x6d3ed0, gensteps=0x8b2b800, cfg4evt=true) at /home/blyth/opticks/optickscore/Opticks.cc:1330
    #11 0x00007ffff7bafcbb in OKG4Mgr::propagate_ (this=0x7fffffff3f10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:214
    #12 0x00007ffff7bafb8d in OKG4Mgr::propagate (this=0x7fffffff3f10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:157
    #13 0x00000000004038c9 in main (argc=33, argv=0x7fffffff4258) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:28
    (gdb) 


    (gdb) f 13
    #13 0x00000000004038c9 in main (argc=33, argv=0x7fffffff5c98) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:28
    28	    okg4.propagate();
    (gdb) f 12
    #12 0x00007ffff7bafb8d in OKG4Mgr::propagate (this=0x7fffffff5950) at /home/blyth/opticks/okg4/OKG4Mgr.cc:157
    157	            propagate_();
    (gdb) f 11
    #11 0x00007ffff7bafcbb in OKG4Mgr::propagate_ (this=0x7fffffff5950) at /home/blyth/opticks/okg4/OKG4Mgr.cc:214
    214	         m_ok->createEvent(gs, cfg4evt);
    (gdb) f 10
    #10 0x00007fffecc8e8e9 in Opticks::createEvent (this=0x6d3ed0, gensteps=0x8b2b5c0, cfg4evt=true) at /home/blyth/opticks/optickscore/Opticks.cc:1330
    1330	    m_run->createEvent(gensteps, cfg4evt );
    (gdb) f 9
    #9  0x00007fffecc7b943 in OpticksRun::createEvent (this=0x6f4ee0, gensteps=0x8b2b5c0, cfg4evt=true) at /home/blyth/opticks/optickscore/OpticksRun.cc:96
    96	    createEvent(tagoffset, cfg4evt ); 
    (gdb) f 8
    #8  0x00007fffecc7ba1d in OpticksRun::createEvent (this=0x6f4ee0, tagoffset=0, cfg4evt=true) at /home/blyth/opticks/optickscore/OpticksRun.cc:111
    111	    m_evt = m_ok->makeEvent(true, tagoffset) ;
    (gdb) f 7
    #7  0x00007fffecc96025 in Opticks::makeEvent (this=0x6d3ed0, ok=true, tagoffset=0) at /home/blyth/opticks/optickscore/Opticks.cc:3299
    3299	    OpticksEvent* evt = OpticksEvent::Make(ok ? m_spec : m_nspec, tagoffset);
    (gdb) f 6
    #6  0x00007fffecc69c6d in OpticksEvent::Make (spec=0x70bed0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEvent.cc:125
    125	     OpticksEventSpec* offspec = spec->clone(tagoffset);
    (gdb) f 5
    #5  0x00007fffecc62349 in OpticksEventSpec::clone (this=0x70bed0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:106
    106	    const char* tag = getOffsetTag(tagoffset);  
    (gdb) f 4
    #4  0x00007fffecc622b1 in OpticksEventSpec::getOffsetTag (this=0x70bed0, tagoffset=0) at /home/blyth/opticks/optickscore/OpticksEventSpec.cc:96
    96	    assert( !iszero && "--tag 0 NOT ALLOWED : AS USING G4 NEGATED CONVENTION " );
    (gdb) 


    084 const char* OpticksEventSpec::getOffsetTag(unsigned tagoffset) const
     85 {
     86     int itag = getITag();
     87     bool iszero = itag == 0 ;
     88     if( iszero )
     89     {
     90         LOG(fatal)
     91             << " iszero itag "
     92             << brief()
     93             ;
     94     }
     95 
     96     assert( !iszero && "--tag 0 NOT ALLOWED : AS USING G4 NEGATED CONVENTION " );
     97     int ntag = itag > 0 ? itag + tagoffset : itag - tagoffset ;
     98     const char* tag = BStr::itoa( ntag );
     99     return tag ;
    100 }

    110 int OpticksEventSpec::getITag() const
    111 {
    112     return m_itag ;
    113 }

    040 OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec)
     41     :
     42     m_pfx(strdup(spec->getPfx())),
     43     m_typ(strdup(spec->getTyp())),
     44     m_tag(strdup(spec->getTag())),
     45     m_det(strdup(spec->getDet())),
     46     m_cat(spec->getCat() ? strdup(spec->getCat()) : NULL),
     47     m_udet(spec->getUDet() ? strdup(spec->getUDet()) : NULL),
     48     m_dir(NULL),
     49     m_reldir(NULL),
     50     m_fold(NULL),
     51     m_itag(spec->getITag())
     52 {
     53 }
     54 
     55 OpticksEventSpec::OpticksEventSpec(const char* pfx, const char* typ, const char* tag, const char* det, const char* cat)
     56     :
     57     m_pfx(strdup(pfx)),
     58     m_typ(strdup(typ)),
     59     m_tag(strdup(tag)),
     60     m_det(strdup(det)),
     61     m_cat(cat ? strdup(cat) : NULL),
     62     m_udet(cat && strlen(cat) > 0 ? strdup(cat) : strdup(det)),
     63     m_dir(NULL),
     64     m_reldir(NULL),
     65     m_fold(NULL),
     66     m_itag(BStr::atoi(m_tag, 0))
     67 {
     68 }

    2431 void Opticks::defineEventSpec()
    2432 {
    2433     const char* cat = m_cfg->getEventCat(); // expected to be defined for tests and equal to the TESTNAME from bash functions like tboolean-
    2434     const char* udet = getInputUDet();
    2435     const char* tag = m_cfg->getEventTag();
    2436     const char* ntag = BStr::negate(tag) ;
    2437     const char* typ = getSourceType();
    2438 
    2439     const char* resource_pfx = m_rsc->getEventPfx() ;
    2440     const char* config_pfx = m_cfg->getEventPfx() ;
    2441     const char* pfx = config_pfx ? config_pfx : resource_pfx ;
    2442     if( !pfx )
    2443     {
    2444         pfx = DEFAULT_PFX ;
    2445         LOG(fatal)
    2446             << " resource_pfx " << resource_pfx
    2447             << " config_pfx " << config_pfx
    2448             << " pfx " << pfx
    2449             << " cat " << cat
    2450             << " udet " << udet
    2451             << " typ " << typ
    2452             << " tag " << tag
    2453             ;
    2454     }
    2455     //assert( pfx ); 
    2456 
    2457 
    2458     m_spec  = new OpticksEventSpec(pfx, typ,  tag, udet, cat );
    2459     m_nspec = new OpticksEventSpec(pfx, typ, ntag, udet, cat );
    2460 
    2461     LOG(LEVEL)
    2462          << " pfx " << pfx
    2463          << " typ " << typ
    2464          << " tag " << tag
    2465          << " ntag " << ntag
    2466          << " udet " << udet
    2467          << " cat " << cat
    2468          ;
    2469 
    2470 }


    1515 template <class Listener>
    1516 const char* OpticksCfg<Listener>::getEventTag() const
    1517 {
    1518     return m_event_tag.empty() ? NULL : m_event_tag.c_str() ;
    1519 }
    1520 template <class Listener>
    1521 const char* OpticksCfg<Listener>::getEventCat() const
    1522 {
    1523     const char* cat_envvar_default = SSys::getenvvar("TESTNAME" , NULL );
    1524     return m_event_cat.empty() ? cat_envvar_default : m_event_cat.c_str() ;
    1525 }
    1526 template <class Listener>
    1527 const char* OpticksCfg<Listener>::getEventPfx() const
    1528 {
    1529     const char* pfx_envvar_default = SSys::getenvvar("TESTNAME" , NULL );
    1530     return m_event_pfx.empty() ? pfx_envvar_default : m_event_pfx.c_str() ;
    1531 }
    1532 



    0916    m_desc.add_options()
     917        ("tag",   boost::program_options::value<std::string>(&m_event_tag), "non zero positive integer string identifying an event" );
     918 
     919    m_desc.add_options()
     920        ("itag",   boost::program_options::value<std::string>(&m_integrated_event_tag), "integrated eventtag to load/save, used from OPG4 package" );
     921 


     756 tboolean--(){
     757 
     758     #tboolean-
     759 
     760     local msg="=== $FUNCNAME :"
     761     local cmdline=$*
     762 
     763     local stack=2180  # default
     764 
     765     local testconfig=$(tboolean-testconfig)
     766     local torchconfig=$(tboolean-torchconfig)
     767 
     768     tboolean-info
     769     [ "$testconfig" == "" ] && echo $msg no testconfig : try ${testname}- && return 255
     770 
     771     o.sh  \
     772             $cmdline \
     773             --envkey \
     774             --rendermode +global,+axis \
     775             --geocenter \
     776             --stack $stack \
     777             --eye $(tboolean-eye) \
     778             --up $(tboolean-up) \
     779             --test \
     780             --testconfig "$testconfig" \
     781             --torch \
     782             --torchconfig "$torchconfig" \
     783             --torchdbg \
     784             --tag $(tboolean-tag) \
     785             --anakey tboolean \
     786             --args \
     787             --save
     788 

     523 tboolean-tag(){  echo ${TAG:-1} ; }
     524 tboolean-det(){  echo boolean ; }
     525 tboolean-src(){  echo torch ; }
     526 tboolean-args(){ echo  --det $(tboolean-det) --src $(tboolean-src) ; }


::

    O[blyth@localhost tests]$ echo $TAG
    O



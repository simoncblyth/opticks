output_event_directory_control_by_envvar
=========================================

issue : control of event paths
-----------------------------------

tds3ip events clearly belong in a different tree, currently 
they are stomping on ordinary running events, eg::

    /tmp/blyth/opticks/source/evt/g4live/natural/1/20210607_181128/OpticksProfileAccLabels.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/1/20210607_181128/OpticksProfileAcc.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/1/20210607_181128/OpticksProfile.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/so.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/rx.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/ox.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/hy.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/ht.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/ph.npy
    /tmp/blyth/opticks/source/evt/g4live/natural/-1/idom.npy


How to change event save paths ?
------------------------------------

::

    1929 void OpticksEvent::saveHitData(NPY<float>* ht) const
    1930 {
    1931     if(ht)
    1932     {
    1933         unsigned num_hit = ht->getNumItems();
    1934         ht->save(m_pfx, "ht", m_typ,  m_tag, m_udet);  // even when zero hits
    1935         LOG(LEVEL)
    1936              << " num_hit " << num_hit
    1937              << " ht " << ht->getShapeString()
    1938              << " tag " << m_tag
    1939              ;
    1940     }
    1941 }

    1991 void OpticksEvent::saveGenstepData()
    1992 {
    1993     // genstep were formally not saved as they exist already elsewhere,
    1994     // however recording the gs in use for posterity makes sense
    1995     // 
    1996     NPY<float>* gs = getGenstepData();
    1997     if(gs) gs->save(m_pfx, "gs", m_typ,  m_tag, m_udet);
    1998 }
    1999 void OpticksEvent::savePhotonData()
    2000 {
    2001     NPY<float>* ox = getPhotonData();
    2002     if(ox) ox->save(m_pfx, "ox", m_typ,  m_tag, m_udet);
    2003 }


Nasty, still using NPY functionality (that should not really belong in NPY package)::

    1089 template <typename T>
    1090 void NPY<T>::save(const char* pfx, const char* tfmt, const char* source, const char* tag, const char* det) const
    1091 {
    1092     std::string path_ = BOpticksEvent::path(pfx, det, source, tag, tfmt );
    1093     save(path_.c_str());
    1094 }
    1095 




::

    076 /**
     77 BOpticksEvent::directory_
     78 ----------------------------
     79 
     80 pfx 
     81     highest level directory name, eg "source"  
     82 
     83 top (geometry)
     84     old and new: BoxInBox,PmtInBox,dayabay,prism,reflect,juno,... 
     85 
     86 sub 
     87     old: cerenkov,oxcerenkov,oxtorch,txtorch   (constituent+source)
     88     new: cerenkov,scintillation,natural,torch  (source only)
     89     
     90 tag
     91     old: tag did not contribute to directory 
     92     
     93 anno
     94     normally NULL, used for example with metadata for a timestamp folder
     95     within the tag folder
     96 
     97 
     98 When a tag is provided the DEFAULT_DIR_TEMPLATE yields::
     99 
    100     $OPTICKS_EVENT_BASE/{pfx}/evt/{top}/{sub}/{tag}
    101 
    102 This is appended with ANNO when that is provided
    103 
    104 
    105 **/
    106 
    107 std::string BOpticksEvent::directory(const char* pfx, const char* top, const char* sub, const char* tag, const char* anno)
    108 {
    109     bool notag = tag == NULL ;
    110     std::string base = directory_template(notag);
    111     std::string base0 = base ;
    112 
    113     replace(base, pfx, top, sub, tag) ;
    114 
    115     std::string dir = BFile::FormPath( base.c_str(), anno  );




The OpticksEventSpec provides the inputs to forming the directory::

     40 OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec)
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


Where do they come from::

    3760 OpticksEvent* Opticks::makeEvent(bool ok, unsigned tagoffset)
    3761 {
    3762     setTagOffset(tagoffset) ;
    3763 
    3764     OpticksEvent* evt = OpticksEvent::Make(ok ? m_spec : m_nspec, tagoffset);
    3765 
    3766     evt->setId(m_event_count) ;   // starts from id 0 
    3767     evt->setOpticks(this);
    3768     evt->setEntryCode(getEntryCode());
    3769 
    3770     LOG(LEVEL)
    3771         << ( ok ? " OK " : " G4 " )
    3772         << " tagoffset " << tagoffset
    3773         << " id " << evt->getId()
    3774         ;
    3775 
    3776     m_event_count += 1 ;
    3777 
    3778 
    3779     const char* x_udet = getEventDet();
    3780     const char* e_udet = evt->getUDet();
    3781 


    2753 /**
    2754 Opticks::defineEventSpec
    2755 -------------------------
    2756 
    2757 Invoked from Opticks::configure after commandline parse and initResource.
    2758 The components of the spec determine file system paths of event files.
    2759 
    2760 
    2761 OpticksCfg::m_event_pfx "--pfx"
    2762    event prefix for organization of event files, typically "source" or the name of 
    2763    the creating executable or the testname 
    2764 
    2765 OpticksCfg::m_event_cat "--cat" 
    2766    event category for organization of event files, typically used instead of detector 
    2767    for test geometries such as prism and lens, default ""
    2768 
    2769 OpticksCfg::m_event_tag "--tag"
    2770    event tag, non zero positive integer string identifying an event 
    2771 
    2772 
    2773 
    2774 **/
    2775 
    2776 const char* Opticks::DEFAULT_PFX = "default_pfx" ;
    2777 
    2778 void Opticks::defineEventSpec()
    2779 {   
    2780     const char* cat = m_cfg->getEventCat(); // expected to be defined for tests and equal to the TESTNAME from bash functions like tboolean-
    2781     const char* udet = getInputUDet(); 
    2782     const char* tag = m_cfg->getEventTag();
    2783     const char* ntag = BStr::negate(tag) ;
    2784     const char* typ = getSourceType();
    2785     
    2786     const char* resource_pfx = m_rsc->getEventPfx() ;
    2787     const char* config_pfx = m_cfg->getEventPfx() ; 
    2788     const char* pfx = config_pfx ? config_pfx : resource_pfx ;
    2789     if( !pfx )
    2790     {   
    2791         pfx = DEFAULT_PFX ;
    2792         LOG(fatal) 
    2793             << " resource_pfx " << resource_pfx
    2794             << " config_pfx " << config_pfx
    2795             << " pfx " << pfx
    2796             << " cat " << cat
    2797             << " udet " << udet
    2798             << " typ " << typ
    2799             << " tag " << tag
    2800             ;
    2801     }
    2802     //assert( pfx ); 
    2803 
    2804     
    2805     m_spec  = new OpticksEventSpec(pfx, typ,  tag, udet, cat );
    2806     m_nspec = new OpticksEventSpec(pfx, typ, ntag, udet, cat );
    2807     
    2808     LOG(LEVEL) 
    2809          << " pfx " << pfx
    2810          << " typ " << typ
    2811          << " tag " << tag
    2812          << " ntag " << ntag
    2813          << " udet " << udet
    2814          << " cat " << cat
    2815          ;
    2816 
    2817 }



The pfx comes either from config_pfx or resource_pfx, with config_pfx overriding::

     781 
     782 void BOpticksResource::initViaKey()
     783 {
     ...
     897 
     898     // see notes/issues/opticks-event-paths.rst 
     899     // matching python approach to event path addressing 
     900     // aiming to eliminate srcevtbase and make evtbase mostly constant
     901     // and equal to idpath normally and OPTICKS_EVENT_BASE eg /tmp for test running
     902     //
     903     // KeySource means name of current executable is same as the one that created the geocache
     904     m_evtpfx = isKeySource() ? "source" : exename ;
     905     m_res->addName("evtpfx", m_evtpfx );
     906 
     907 

The resource_pfx is where teh default "source" comes from.






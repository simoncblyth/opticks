U4RecorderTest-shakedown
===========================

What Next ?
-------------


* need more featureful geometry to test/develop things like microstep skipping 

  * before full geometry prep a local simple Raindrop geometry 
  * need water and air 



Geant4 originals : expand from just LS_ori to all materials 
--------------------------------------------------------------

::

    0805 void GPropertyLib::addRawOriginal(GPropertyMap<double>* pmap)
     806 {
     807     m_raw_original.push_back(pmap);
     808 }
     ...
     845 GPropertyMap<double>* GPropertyLib::getRawOriginal(const char* shortname) const
     846 {
     847     unsigned num_raw_original = m_raw_original.size();
     848     for(unsigned i=0 ; i < num_raw_original ; i++)
     849     { 
     850         GPropertyMap<double>* pmap = m_raw_original[i];
     851         const char* name = pmap->getShortName();
     852         if(strcmp(shortname, name) == 0) return pmap ;
     853     }
     854     return NULL ;
     855 }

    epsilon:ggeo blyth$ opticks-f addRawOriginal
    ./extg4/X4PhysicalVolume.cc:        m_sclib->addRawOriginal(pmap);      
    ./extg4/X4MaterialTable.cc:        m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib
    ./ggeo/GPropertyLib.cc:void GPropertyLib::addRawOriginal(GPropertyMap<double>* pmap)
    ./ggeo/GPropertyLib.hh:        void                  addRawOriginal(GPropertyMap<double>* pmap);
    epsilon:opticks blyth$ 


     342 void X4PhysicalVolume::collectScintillatorMaterials()
     343 {
     ...
     348     typedef GPropertyMap<double> PMAP ;
     349     std::vector<PMAP*> raw_energy_pmaps ;
     350     m_mlib->findRawOriginalMapsWithProperties( raw_energy_pmaps, SCINTILLATOR_PROPERTIES, ',' );
     ...
     378     // original energy domain 
     379     for(unsigned i=0 ; i < num_scint ; i++)
     380     {
     381         PMAP* pmap = raw_energy_pmaps[i] ;
     382         m_sclib->addRawOriginal(pmap);
     383     }

    105 void X4MaterialTable::init()
    106 {
    107     unsigned num_input_materials = m_input_materials.size() ;
    ...
    111     for(unsigned i=0 ; i < num_input_materials ; i++)
    112     {
    ...
    136         char mode_asis_en = 'E' ;
    137         GMaterial* rawmat_en = X4Material::Convert( material, mode_asis_en );
    138         GPropertyMap<double>* pmap_rawmat_en = dynamic_cast<GPropertyMap<double>*>(rawmat_en) ;
    139         m_mlib->addRawOriginal(pmap_rawmat_en) ;  // down to GPropertyLib

    0887 void GPropertyLib::findRawOriginalMapsWithProperties( std::vector<GPropertyMap<double>*>& dst, const char* props, char delim )
     888 {
     889     SelectPropertyMapsWithProperties(dst, props, delim, m_raw_original );
     890 }

    0982 void GPropertyLib::saveRawOriginal()
     983 {
     984     std::string dir = getCacheDir();
     985     unsigned num_raw_original = m_raw_original.size();
     986     LOG(LEVEL) << "[ " << dir << " num_raw_original " << num_raw_original ;
     987     for(unsigned i=0 ; i < num_raw_original ; i++)
     988     {
     989         GPropertyMap<double>* pmap = m_raw_original[i] ;
     990         pmap->save(dir.c_str());
     991     }
     992     LOG(LEVEL) << "]" ;
     993 }

    001 #include "SConstant.hh"
      2 
      3 const char* SConstant::ORIGINAL_DOMAIN_SUFFIX = "_ori" ;
      4 

    1076 template <typename T>
    1077 void GPropertyMap<T>::save(const char* dir)
    1078 {
    1079     std::string shortname = m_shortname ;
    1080     if(m_original_domain) shortname += SConstant::ORIGINAL_DOMAIN_SUFFIX ;
    1081 
    1082     LOG(LEVEL) << " save shortname (+_ori?) [" << shortname << "] m_original_domain " << m_original_domain  ;
    1083 
    1084     for(std::vector<std::string>::iterator it=m_keys.begin() ; it != m_keys.end() ; it++ )
    1085     {
    1086         std::string key = *it ;
    1087         std::string propname(key) ;
    1088         propname += ".npy" ;
    1089 
    1090         GProperty<T>* prop = m_prop[key] ;
    1091         prop->save(dir, shortname.c_str(), propname.c_str());  // dir, reldir, name
    1092     }
    1093 }


geocache-create uses okg4/tests/OKX4Test.cc::

    112     
    113     m_ggeo->postDirectTranslation();   // closing libs, finding repeat instances, merging meshes, saving 
    114     

    0584 /**
     585 GGeo::postDirectTranslation
     586 -------------------------------
     587 
     588 Invoked from G4Opticks::translateGeometry after the X4PhysicalVolume conversion
     589 for live running or from okg4/tests/OKX4Test.cc main for geocache-create.
     590 
     591 **/
     592 
     593 
     594 void GGeo::postDirectTranslation()
     595 {
     596     LOG(LEVEL) << "[" ;
     597 
     598     prepare();     // instances are formed here     
     599 
     600     LOG(LEVEL) << "( GBndLib::fillMaterialLineMap " ;
     601     GBndLib* blib = getBndLib();
     602     blib->fillMaterialLineMap();
     603     LOG(LEVEL) << ") GBndLib::fillMaterialLineMap " ;
     604 
     605     LOG(LEVEL) << "( GGeo::save " ;
     606     save();
     607     LOG(LEVEL) << ") GGeo::save " ;
     608 
     609 
     610     deferred();
     611 
     612     postDirectTranslationDump();
     613 
     614     LOG(LEVEL) << "]" ;
     615 }


With Gun : First 100 label id are zero ? FIXED 
------------------------------------------------

::

    In [25]: np.all( id_[100:] == np.arange(100,388, dtype=np.int32)  )
    Out[25]: True

    In [26]: np.all( id_[:100] == 0 )
    Out[26]: True

FIXED by commenting the SEvt::AddTorchGenstep when gun running::

    133 int main(int argc, char** argv)
    134 {    
    135     OPTICKS_LOG(argc, argv);
    136 
    137     unsigned max_bounce = 9 ;
    138     SEventConfig::SetMaxBounce(max_bounce);
    139     SEventConfig::SetMaxRecord(max_bounce+1);
    140     SEventConfig::SetMaxRec(max_bounce+1);
    141     SEventConfig::SetMaxSeq(max_bounce+1);
    142 
    143     SEvt evt ; 
    144     //SEvt::AddTorchGenstep();


With Gun : FIXED : Unexpected seq labels 
-----------------------------------------

* should be starting with SI or CK 

::

   0 : MI SD SD SD MI MI 
   1 : MI SD SD SD MI MI 
   2 : MI SD SD MI MI MI 
   3 : MI SD SD MI MI MI 
   4 : MI SC SD MI MI MI 
   5 : SI SC SD MI MI MI 
   6 : SI SC SD MI MI MI 
   7 : SI AB AB MI 
   8 : SI AB AB MI 


After zeroing seq and rec at SEvt::startPhoton the seq looks more reasonable::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 


With Gun : Not terminated at AB ? Probably reemision rejoin AB scrub not working yet ? YEP: FIXED
----------------------------------------------------------------------------------------------------

* actually did i implement that at all ? only did the flagmask not the seqhis ?

seqhis::

   0 : CK AB AB 
   1 : CK AB SC AB MI 
   2 : CK AB 
   3 : CK MI 
   4 : CK AB 
   5 : SI AB 
   6 : SI SC MI MI MI MI 
   7 : SI AB 
   8 : SI AB AB MI 
   9 : SI MI 

Implement GIDX control for debug running with single genstep.::

    bflagdesc_(r[0,j])
     idx(     0) prd(  0    0     0 0 ii:    0)  CK               CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB            AB|CK  
     idx(     0) prd(  0    0     0 0 ii:    0)  AB         RE|AB|CK  


* FIXED : clear discrepancy between the flag+seqhis and the flagmask 

The current_photon flag gets seq.add_nibble by SEvt::pointPhoton::

    342 void SEvt::pointPhoton(const spho& label)
    343 {   
    344     assert( label.isSameLineage(current_pho) );
    345     unsigned idx = label.id ;
    346     int& bounce = slot[idx] ;
    347     
    348     const sphoton& p = current_photon ;
    349     srec& rec = current_rec ;
    350     sseq& seq = current_seq ;
    351     
    352     if( evt->record && bounce < evt->max_record ) evt->record[evt->max_record*idx+bounce] = p ;
    353     if( evt->rec    && bounce < evt->max_rec    ) evt->add_rec(rec, idx, bounce, p );  
    354     if( evt->seq    && bounce < evt->max_seq    ) seq.add_nibble(bounce, p.flag(), p.boundary() );
    355     
    356     bounce += 1 ;
    357 }

Fixed reemission bookkeeping by history rewrite.

SEvt::rjoinPhoton::


    331     if( evt->photon )
    332     {
    333        // HMM: could directly change photon[idx] via ref ? 
    334        // But are here taking a copy to current_photon
    335        // and relying on copyback at SEvt::endPhoton
    336 
    337         current_photon = photon[idx] ;
    338         assert( current_photon.flag() == BULK_ABSORB );
    339         assert( current_photon.flagmask & BULK_ABSORB  );   // all continuePhoton should have BULK_ABSORB in flagmask
    340 
    341         current_photon.flagmask &= ~BULK_ABSORB  ; // scrub BULK_ABSORB from flagmask
    342         current_photon.set_flag(BULK_REEMIT) ;     // gets OR-ed into flagmask 
    343     }
    344 
    345     if( evt->seq )
    346     {
    347         current_seq = seq[idx] ;
    348         unsigned seq_flag = current_seq.get_flag(prior);
    349         assert( seq_flag == BULK_ABSORB );
    350         current_seq.set_flag(prior, BULK_REEMIT);
    351     }
    352 
    353     if( evt->record )
    354     {
    355         sphoton& rjoin_record = evt->record[evt->max_record*idx+prior]  ;
    356         unsigned rjoin_flag = rjoin_record.flag() ;
    357 
    358         LOG(info) << " rjoin.flag "  << OpticksPhoton::Flag(rjoin_flag)  ;
    359         assert( rjoin_flag == BULK_ABSORB );
    360         assert( rjoin_record.flagmask & BULK_ABSORB );
    361 
    362         rjoin_record.flagmask &= ~BULK_ABSORB ; // scrub BULK_ABSORB from flagmask  
    363         rjoin_record.set_flag(BULK_REEMIT) ;
    364     }


GIDX selection beyond the first is asserting : FIXED 
--------------------------------------------------------

::

    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfRunAction@38] 
    2022-06-09 16:52:41.855 INFO  [19428647] [U4Recorder::BeginOfEventAction@40] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     5 off      0 typ G4Cerenkov_modified gidx 0 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   0 pho     1 off      0 typ DsG4Scintillation_r4695 gidx 1 enabled 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::setNumPhoton@210]  numphoton 1
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 2 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 3 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::addGenstep@183]  s.desc sgs: idx   1 pho     1 off      1 typ DsG4Scintillation_r4695 gidx 4 enabled 0
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@269] 
    2022-06-09 16:52:41.856 INFO  [19428647] [SEvt::beginPhoton@270] spho ( gs ix id gn   1   0    1 0 ) 
    2022-06-09 16:52:41.856 ERROR [19428647] [SEvt::beginPhoton@275]  not in_range  idx 1 pho.size  1 label spho ( gs ix id gn   1   0    1 0 ) 
    Assertion failed: (in_range), function beginPhoton, file /Users/blyth/opticks/sysrap/SEvt.cc, line 281.
    ./U4RecorderTest.sh: line 43: 73818 Abort trap: 6           U4RecorderTest
    === ./U4RecorderTest.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    epsilon:tests blyth$ 



The sgs genstep labelling is using an offset that does not account for enabled gensteps presumably::

     56 inline spho sgs::MakePho(unsigned idx, const spho& ancestor)
     57 {
     58     return ancestor.isDefined() ? ancestor.make_reemit() : spho::MakePho(index, idx, offset + idx, 0) ;
     59 }


FIXED this by simplifying genstep disabling to simply set the numphotons of disabled gensteps to zero, 
without any change to the collection machinery.  As genstep disabling is purely for debugging this is acceptable. 




FIXED : Checking rjoinPhoton matching tripping some asserts
---------------------------------------------------------------


::

    u4 ; cd tests

    epsilon:tests blyth$ ./U4RecorderTest.sh


    2022-06-09 20:51:29.134 INFO  [19769941] [SEvt::rjoinPhoton@315] 
    2022-06-09 20:51:29.134 INFO  [19769941] [SEvt::rjoinPhoton@316] spho ( gs ix id gn 117   0  33310 ) 
    rjoinPhotonCheck : does not have BULK_ABSORB flag ? ph.idx 333 flag_AB NO flagmask_AB NO
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 4 id 0 or 1.000 ix 333 fm 16 ab MI
     digest(16) 1bf2798f0385a6f99531161605e3e661
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
     NOT seq_flag_AB, rather   
     rjoin_record_d12   1e80c7b62fe41f2b3cfbc743988d1787
     current_photon_d12 62c0957fc9dbf3ed296559467aa5d5d5
     d12_match NO
    Assertion failed: (d12_match), function rjoinPhoton, file /Users/blyth/opticks/sysrap/SEvt.cc, line 377.
    ./U4RecorderTest.sh: line 43: 23381 Abort trap: 6           U4RecorderTest
    === ./U4RecorderTest.sh : logdir /tmp/blyth/opticks/U4RecorderTest
    /Users/blyth/opticks/u4/tests
    cfbase:/usr/local/opticks/geocache/OKX4Test_lWorld0x5780b30_PV_g4live/g4ok_gltf/5303cd587554cb16682990189831ae83/1/CSG_GGeo 
    Fold : setting globals False globals_prefix  
    t



FIXED : Smoking gun is getting impossible rjoin.flag of SCINTILLATION are clearly 
wandering over to another photons records::

    2022-06-10 11:56:09.859 INFO  [19958285] [SEvt::rjoinPhoton@321] 
    2022-06-10 11:56:09.859 INFO  [19958285] [SEvt::rjoinPhoton@322] spho (gs:ix:id:gn 117   0    0 10)
    rjoinPhotonCheck : does not have BULK_ABSORB flag ? sphoton idx 0 flag MISS flagmask SI|MI|RE
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 4 id 0 or 1.000 ix 0 fm 16 ab MI
     digest(16) 7706526a21ed79f8fb759805c75c798b
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
     NOT seq_flag_AB, rather   
     idx 0 bounce 11 prior 10 evt.max_record 10 rjoin_record_d12   1e80c7b62fe41f2b3cfbc743988d1787
     current_photon_d12 62c0957fc9dbf3ed296559467aa5d5d5
     d12match NO
     rjoin_record 
     pos (-9.399,42.455,114.610)  t  7.007
     mom ( 0.802, 0.597, 0.017)  iindex 0
     pol ( 0.559,-0.739,-0.377)  wl 466.605
     bn 0 fl 2 id 0 or 1.000 ix 1 fm 2 ab SI
     digest(16) 07cb368115014bb1c643bd028d48c1e0
     digest(12) 1e80c7b62fe41f2b3cfbc743988d1787
    2022-06-10 11:56:09.860 INFO  [19958285] [SEvt::rjoinPhoton@400]  rjoin.flag SCINTILLATION
     NOT rjoin_flag_AB 
     NOT rjoin_record_flagmask_AB 
     current_photon 
     pos (-1000.000,722.148,670.385)  t  46.844
     mom (-0.814, 0.581,-0.026)  iindex 0
     pol (-0.145,-0.159, 0.977)  wl 394.830
     bn 0 fl 10 id 0 or 1.000 ix 0 fm 16 ab RE
     digest(16) 829c294403eff470277c9cdb81f983a6
     digest(12) 62c0957fc9dbf3ed296559467aa5d5d5
    2022-06-10 11:56:09.860 INFO  [19958285] [SEvt::pointPhoton@494] spho (gs:ix:id:gn 117   0    0 10)  seqhis      55555555552 nib 11 SI RE RE RE RE RE RE RE RE RE RE                
    2022-06-10 11:56:09.860 INFO  [19958285] [U4Recorder::UserSteppingAction_Optical@190]  step.tstat fStopAndKill MISS



Must review how evt->max_record truncation is handled, as apparently not working.

* FIXED : the problem was just with the rjoin checking not applying the truncation








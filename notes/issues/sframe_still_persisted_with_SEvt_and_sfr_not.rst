sframe_still_persisted_with_SEvt_and_sfr_not
===============================================


Motivation
-----------

* MortonOverlapScan_test.sh needs to know the MOI frame bbox used by simtrace creation of point clouds
* currently SEvt still saving the old sframe

Issue
-------

* sframe sfr duplication 


Where the sframe::save happens
--------------------------------

::

    BP=sframe::getFrameArray ./cxt_min.sh

    (gdb) bt
    #0  sframe::getFrameArray (this=0x159790c0) at /home/blyth/opticks/sysrap/sframe.h:614
    #1  0x00007ffff558e3c7 in sframe::save (this=0x159790c0, dir=0x1b4a4f00 "/data1/blyth/tmp/GEOM/J25_7_2_opticks_Debug/CSGOptiXTMTest/ALL0_2dxy/A000", name_=0x7ffff5742e2f "sframe") at /home/blyth/opticks/sysrap/sframe.h:637
    #2  0x00007ffff5578837 in SEvt::saveFrame (this=0x15978f40, dir=0x1b4a4f00 "/data1/blyth/tmp/GEOM/J25_7_2_opticks_Debug/CSGOptiXTMTest/ALL0_2dxy/A000") at /home/blyth/opticks/sysrap/SEvt.cc:4640
    #3  0x00007ffff5577f05 in SEvt::save (this=0x15978f40, dir_=0x7ffff570e5c8 "$TMP/GEOM/$GEOM/$ExecutableName") at /home/blyth/opticks/sysrap/SEvt.cc:4537
    #4  0x00007ffff55771ac in SEvt::save (this=0x15978f40) at /home/blyth/opticks/sysrap/SEvt.cc:4384
    #5  0x00007ffff556b0e1 in SEvt::endOfEvent (this=0x15978f40, eventID=0) at /home/blyth/opticks/sysrap/SEvt.cc:1847
    #6  0x00007ffff5e75aee in QSim::simtrace (this=0x18065860, eventID=0) at /home/blyth/opticks/qudarap/QSim.cc:887
    #7  0x00007ffff7e32d9f in CSGOptiX::simtrace (this=0x18079ca0, eventID=0) at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:822
    #8  0x00007ffff7e2f53e in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:157
    #9  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb248) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 


::

    4637 void SEvt::saveFrame(const char* dir) const
    4638 {
    4639     LOG(LEVEL) << "[ dir " << dir ;
    4640     frame.save(dir);
    4641     LOG(LEVEL) << "] dir " << dir ;
    4642 }


Where SEvt::setFrame comes from
--------------------------------

::

    BP=SEvt::setFrame ./cxt_min.sh

    (gdb) bt
    #0  SEvt::setFrame (this=0x15978f40, fr=...) at /home/blyth/opticks/sysrap/SEvt.cc:797
    #1  0x00007ffff55699b0 in SEvt::SetFrame (fr=...) at /home/blyth/opticks/sysrap/SEvt.cc:1406
    #2  0x00007ffff7685b25 in CSGFoundry::AfterLoadOrCreate () at /home/blyth/opticks/CSG/CSGFoundry.cc:3867
    #3  0x00007ffff7682dff in CSGFoundry::Load () at /home/blyth/opticks/CSG/CSGFoundry.cc:3168
    #4  0x00007ffff7e2f50e in CSGOptiX::SimtraceMain () at /home/blyth/opticks/CSGOptiX/CSGOptiX.cc:155
    #5  0x0000000000404a95 in main (argc=1, argv=0x7fffffffb258) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXTMTest.cc:13
    (gdb) 


::

    3856 void CSGFoundry::AfterLoadOrCreate() // static
    3857 {
    3858     CSGFoundry* fd = CSGFoundry::Get();
    3859 
    3860     SEvt::CreateOrReuse() ;   // creates 1/2 SEvt depending on OPTICKS_INTEGRATION_MODE
    3861 
    3862     if(!fd) return ;
    3863 
    3864     sframe fr = fd->getFrameE() ;
    3865 
    3866     LOG(LEVEL) << fr ;
    3867     SEvt::SetFrame(fr); // now only needs to be done once to transform input photons
    3868 }
    3869 



SGeo base for CSGFoundry
--------------------------

::

     220 /**
     221 CSGOptiX::InitEvt  TODO : THIS DOES NOT USE GPU : SO SHOULD BE ELSEWHERE
     222 --------------------------------------------------------------------------
     223 
     224 Invoked from CSGOptiX::Create
     225 
     226 
     227 Q: Why the SEvt geometry connection ?
     228 A: Needed for global to local transform conversion
     229 
     230 Q: What uses SEVt::setGeo (SGeo) ?
     231 A: Essential set_matline of Cerenkov Genstep
     232 
     233 **/
     234 
     235 void CSGOptiX::InitEvt( CSGFoundry* fd  )
     236 {
     237     SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ;
     238 
     239     sev->setGeo((SGeo*)fd);
     240 
     241     std::string* rms = SEvt::RunMetaString() ;
     242     assert(rms);
     243 
     244     bool stamp = false ;
     245     smeta::Collect(*rms, "CSGOptiX__InitEvt", stamp );
     246 }



Review usage of sft.h sframe.h by simtrace running including the cxt_min.py
--------------------------------------------------------------------------------

stree::get_frame methods mostly use sfr::

     507    // transitional method for matching with CSGFoundry::getFrame
     508     void get_frame_f4( sframe& fr, int idx ) const ;
     509 
     510 
     511     sfr  get_frame_moi() const ;
     512 
     513     sfr  get_frame_extent(const char* s_extent ) const ;
     514     sfr  get_frame_axis(const char* s_axis ) const ;
     515     sfr  get_frame_prim(const char* s_prim ) const ;
     516     sfr  get_frame_nidx(const char* s_nidx ) const ;
     517     sfr  get_frame_prim(int prim ) const ;
     518     sfr  get_frame_nidx(int nidx ) const ;
     519 
     520     sfr  get_frame(const char* q_spec) const ;
     521     int  get_frame_from_npyfile(sfr& f, const char* q_spec ) const ;
     522     int  get_frame_from_triplet(sfr& f, const char* q_spec ) const ;
     523     int  get_frame_from_coords( sfr& f, const char* q_spec ) const ;
     524     int  get_frame_from_transform( sfr& f, const char* q_spec ) const ;
     525 
     526     bool has_frame(const char* q_spec) const ;
     527 
     528 
     529     int get_frame_instanced(  sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal, std::ostream* out = nullptr, VTR* t_stack = nullptr ) const ;
     530     int get_frame_remainder(  sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
     531     int get_frame_triangulate(sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
     532     int get_frame_global(     sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal ) const ;
     533     int _get_frame_global(     sfr& f, int lvid, int lvid_ordinal, int repeat_ordinal, char ridx_type ) const ;


cxt_min.sh not calling stree::get_frame_moi
---------------------------------------------

::

    BP=stree::get_frame_moi ./cxt_min.sh


cxr_min.sh does call stree::get_frame_moi
--------------------------------------------

::

    (gdb) bt
    #0  stree::get_frame_moi (this=0x542600) at /data1/blyth/local/opticks_Debug/include/SysRap/stree.h:2239
    #1  0x000000000048b012 in SGLM::setTreeScene (this=0x1bd95e80, _tree=0x542600, _scene=0x109cb170) at /data1/blyth/local/opticks_Debug/include/SysRap/SGLM.h:877
    #2  0x0000000000495ea5 in CSGOptiXRenderInteractiveTest::init (this=0x7fffffffad50) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:124
    #3  0x0000000000495d94 in CSGOptiXRenderInteractiveTest::CSGOptiXRenderInteractiveTest (this=0x7fffffffad50) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:113
    #4  0x00000000004465cf in main (argc=1, argv=0x7fffffffaef8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderInteractiveTest.cc:206
    (gdb) 

::

    116 inline void CSGOptiXRenderInteractiveTest::init()
    117 {
    118     assert( irc == 0 );
    119     assert(fd);
    120     stree* tree = fd->getTree();
    121     assert(tree);
    122     SScene* scene = fd->getScene() ;
    123     assert(scene);
    124     gm->setTreeScene(tree, scene);
    125     gm->setRecord(ar, br);
    126 
    127     cx = CSGOptiX::Create(fd) ;
    128     gl = new SGLFW(*gm);
    129     interop = new SGLFW_CUDA(*gm);
    130     glev    = new SGLFW_Evt(*gl);
    131 
    132     if(gl->level > 0) std::cout << "CSGOptiXRenderInteractiveTest::init before render loop  gl.get_wanted_frame_idx " <<  gl->get_wanted_frame_idx() << "\n" ;
    133     if(level > 0) std::cout << "CSGOptiXRenderInteractiveTest::init [" << _level << "][" << level << "]\n" << desc() ;
    134 
    135 }







material-names-wrong-in-gui
==============================

FIXED ISSUE: some GUI material names wrong for test geometry
---------------------------------------------------------------

::

   tboolean-media --okg4 --load 


Observations
----------------


* seqmat GItemIndex coming from okg-/OpticksIdx is providing labels with wrong materials
  for test geometry running 

* CAUSE : was due to inappropriate direct use of m_ggeo within OpticksGeometry (where m_ggeo is instanciated)
  fixed by getting libs from hub 


GUI tabs
-----------

GUI::show does most of these, with the event specifics done by Photons


* Help
* Params
* Stats
* Interactor
* Scene
* Composition
* View
* Camera
* Clipper
* Trackball
* Bookmarks
* State

* ABOVE ALL DIRECT FROM GUI::show 

* Photon Flag Selection
* Photon Termination Boundaries (Photons.m_boundaries)

  * wrong Vacuum///Vacuum

* Photon Flag Sequence Selection (Photons.m_seqhis)

  * correct

* Photon Material Sequence Selection (Photons.m_seqmat)

  * labels wrong


* BELOW ALL DIRECT FROM GUI::show 

* GMaterialLib

  * correct

* GSurfaceLib

  * correct

* GFlags
* Dev



seqmat indexing
-------------------

::

     68 GItemIndex* OpticksIdx::makeMaterialItemIndex()
     69 {
     70     OpticksEvent* evt = getEvent();
     71     Index* seqmat_ = evt->getMaterialIndex() ;
     72     if(!seqmat_)
     73     {
     74          LOG(warning) << "OpticksIdx::makeMaterialItemIndex NULL seqmat" ;
     75          return NULL ;
     76     }
     77 
     78     OpticksAttrSeq* qmat = m_hub->getMaterialNames();
     79 
     80     GItemIndex* seqmat = new GItemIndex(seqmat_) ;
     81     seqmat->setTitle("Photon Material Sequence Selection");
     82     seqmat->setHandler(qmat);
     83     seqmat->formTable();
     84 
     85     return seqmat ;
     86 }

::

    821 OpticksAttrSeq* OpticksHub::getFlagNames()
    822 {
    823     return m_ok->getFlagNames();
    824 }
    825 OpticksAttrSeq* OpticksHub::getMaterialNames()
    826 {
    827     return m_geometry->getMaterialNames();
    828 }
    829 OpticksAttrSeq* OpticksHub::getBoundaryNames()
    830 {
    831     return m_geometry->getBoundaryNames();
    832 }
    833 std::map<unsigned int, std::string> OpticksHub::getBoundaryNamesMap()
    834 {
    835     return m_geometry->getBoundaryNamesMap();
    836 }


SMOKING GUN : misplaced methods directly accessing m_ggeo within OpticksGeo
-------------------------------------------------------------------------------

::

    098 OpticksAttrSeq* OpticksGeometry::getMaterialNames()
     99 {
    100      OpticksAttrSeq* qmat = m_ggeo->getMaterialLib()->getAttrNames();
    101      qmat->setCtrl(OpticksAttrSeq::SEQUENCE_DEFAULTS);
    102      return qmat ; 
    103 }
    104 
    105 OpticksAttrSeq* OpticksGeometry::getBoundaryNames()
    106 {
    107      GBndLib* blib = m_ggeo->getBndLib();
    108      OpticksAttrSeq* qbnd = blib->getAttrNames();
    109      if(!qbnd->hasSequence())
    110      {    
    111          blib->close();
    112          assert(qbnd->hasSequence());
    113      }    
    114      qbnd->setCtrl(OpticksAttrSeq::VALUE_DEFAULTS);
    115      return qbnd ;
    116 }      
    117        
    118 std::map<unsigned int, std::string> OpticksGeometry::getBoundaryNamesMap()
    119 { 
    120     OpticksAttrSeq* qbnd = getBoundaryNames() ;
    121     return qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;
    122 }  
    123        



 

Review GUI code
------------------
::

    034 OKMgr::OKMgr(int argc, char** argv, const char* argforced )
     35     :
     36     m_log(new SLog("OKMgr::OKMgr")),
     37     m_ok(new Opticks(argc, argv, argforced)),
     38     m_hub(new OpticksHub(m_ok)),            // immediate configure and loadGeometry 
     39     m_idx(new OpticksIdx(m_hub)),
     40     m_num_event(m_ok->getMultiEvent()),     // after hub instanciation, as that configures Opticks
     41     m_gen(m_hub->getGen()),
     42     m_run(m_hub->getRun()),
     43     m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),
     44     m_propagator(new OKPropagator(m_hub, m_idx, m_viz)),
     45     m_count(0)
     46 {
     47     init();
     48     (*m_log)("DONE");
     49 }


    062 OpticksViz::OpticksViz(OpticksHub* hub, OpticksIdx* idx, bool immediate)
     63     :
     64     m_log(new SLog("OpticksViz::OpticksViz")),
     65     m_hub(hub),
     66     m_ok(hub->getOpticks()),
     67     m_run(m_ok->getRun()),
     68     m_ggb(m_hub->getGGeoBase()),
     69     m_idx(idx),


    321 void OpticksViz::indexPresentationPrep()
    322 {
    323     if(!m_idx) return ;
    324 
    325     LOG(info) << "OpticksViz::indexPresentationPrep" ;
    326 
    327     m_seqhis = m_idx->makeHistoryItemIndex();
    328     m_seqmat = m_idx->makeMaterialItemIndex();
    329     m_boundaries = m_idx->makeBoundaryItemIndex();
    330 
    331 }





::

    132 void OpticksViz::visualize()
    133 {
    134     prepareGUI();
    135     renderLoop();
    136 }

    333 void OpticksViz::prepareGUI()
    334 {
    335     Bookmarks* bookmarks=m_hub->getBookmarks();
    336 
    337     bookmarks->create(0);
    338 
    339 #ifdef GUI_
    340 
    341     if(m_idx)
    342     {
    343         Types* types = m_ok->getTypes();  // needed for each render
    344         m_photons = new Photons(types, m_boundaries, m_seqhis, m_seqmat ) ; // GUI jacket 
    345         m_scene->setPhotons(m_photons);
    346     }
    347 
    348     m_gui = new GUI(m_hub) ;
    349     m_gui->setScene(m_scene);
    350     m_gui->setPhotons(m_photons);
    351     m_gui->setComposition(m_hub->getComposition());
    352     m_gui->setBookmarks(bookmarks);
    353     m_gui->setStateGUI(new StateGUI(m_hub->getState()));
    354     m_gui->setInteractor(m_interactor);   // status line


::

     12 class OGLRAP_API Photons {
     13    public:
     14        Photons(Types* types, GItemIndex* boundaries, GItemIndex* seqhis, GItemIndex* seqmat);
     15    public:
     16        void gui();
     17        void gui_flag_selection();
     18        void gui_radio_select(GItemIndex* ii);
     19        void gui_item_index(GItemIndex* ii);
     20        const char* getSeqhisSelectedKey();
     21        const char* getSeqhisSelectedLabel();
     22    private:





::

    043 void Photons::gui()
     44 {
     45 #ifdef GUI_
     46 
     47     if(m_types)
     48     {
     49         ImGui::Spacing();
     50         if (ImGui::CollapsingHeader("Photon Flag Selection"))
     51         {
     52             gui_flag_selection();
     53         }
     54     }
     55 
     56     if(m_boundaries)
     57     {
     58         ImGui::Spacing();
     59         GUI::gui_radio_select(m_boundaries);
     60     }
     61 
     62     if(m_seqhis)
     63     {
     64         ImGui::Spacing();
     65         GUI::gui_radio_select(m_seqhis);
     66     }
     67 
     68     if(m_seqmat)
     69     {
     70         ImGui::Spacing();
     71         GUI::gui_radio_select(m_seqmat);
     72     }
     73 #endif
     74 }


    749 void GUI::gui_radio_select(GItemIndex* ii)
    750 {
    751 #ifdef GUI_
    752     typedef std::vector<std::string> VS ;
    753     Index* index = ii->getIndex();
    754 
    755     if (ImGui::CollapsingHeader(index->getTitle()))
    756     {
    757        VS& labels = ii->getLabels();
    758        VS  names = index->getNames();
    759        assert(names.size() == labels.size());
    760 
    761        int* ptr = index->getSelectedPtr();
    762 
    763        std::string all("All ");
    764        all += index->getItemType() ;
    765 
    766        ImGui::RadioButton( all.c_str(), ptr, 0 );
    767 
    768        for(unsigned int i=0 ; i < labels.size() ; i++)
    769        {
    770            std::string iname = names[i] ;
    771            std::string label = labels[i] ;
    772            unsigned int local  = index->getIndexLocal(iname.c_str()) ;
    773            ImGui::RadioButton( label.c_str(), ptr, local);  // when selected the local value is written into the ptr location
    774        }
    775        ImGui::Text("%s %d ", index->getItemType(), *ptr);
    776    }
    777 #endif
    778 }



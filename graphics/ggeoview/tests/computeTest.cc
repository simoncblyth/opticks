
// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "Composition.hh"
#include "FrameCfg.hh"
// these are here just for commandline handling
//  TODO: move commandline handling out of oglrap- into npy-/ggeoview- ? 


// npy-
#include "NPY.hpp"
#include "G4StepNPY.hpp"
#include "NumpyEvt.hpp"
#include "Types.hpp"
#include "Lookup.hpp"
#include "Timer.hpp"
#include "Parameters.hpp"
#include "Times.hpp"
#include "Report.hpp"
#include "stringutil.hpp"

// ggeo-
#include "GGeo.hh"
#include "GLoader.hh"
#include "GCache.hh"
#include "GMergedMesh.hh" 
class GBoundaryLib ; 



// assimpwrap-
#include "AssimpGGeo.hh"

// optixrap-
#include "OEngine.hh"
#include "RayTraceConfig.hh"


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include "boost/log/utility/setup.hpp"
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void logging_init()
{
   // see blogg-
    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );

    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]: %Message%",
        boost::log::keywords::auto_flush = true
    );  

    boost::log::add_common_attributes();
}


int main(int argc, char** argv)
{
    Parameters p ; 
    Timer t ; 
    t.setVerbose(true);
    t.start();

    logging_init();

    LOG(info) << argv[0] ; 

    GCache cache("GGEOVIEW_") ; 
    const char* idpath = cache.getIdPath();
    const char* det = cache.getDetector();

    Types types ;  
    types.readFlags("$ENV_HOME/graphics/ggeoview/cu/photon.h");

    Frame frame ;
    Cfg cfg("umbrella", false) ; // collect other Cfg objects
    FrameCfg<Frame>* fcfg = new FrameCfg<Frame>("frame", &frame,false);
    cfg.add(fcfg);

    cfg.commandline(argc, argv);
    t.setCommandLine(cfg.getCommandLine()); 

    if(fcfg->hasOpt("idpath")) std::cout << idpath << std::endl ;
    if(fcfg->hasOpt("help"))   std::cout << cfg.getDesc() << std::endl ;
    if(fcfg->isAbort()) exit(EXIT_SUCCESS); 


    unsigned int numcol = 64 ; 
    bool nooptix    = fcfg->hasOpt("nooptix");
    bool nogeocache = false ;
    assert(nooptix == false);

    GLoader loader ;
    loader.setInstanced(false); // find repeated geometry 
    loader.setRepeatIndex(fcfg->getRepeatIndex());
    loader.setTypes(&types);
    loader.setCache(&cache);
    loader.setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr
    loader.load(nogeocache);
    p.add<int>("repeatIdx", loader.getRepeatIndex() );


    GGeo* gg = loader.getGGeo();
    unsigned int nmm = gg->getNumMergedMesh();
    assert(nmm == 1);
    GMergedMesh* mm = gg->getMergedMesh(0); 

    GBoundaryLib* blib = loader.getBoundaryLib();
    Lookup* lookup = loader.getMaterialLookup();

    unsigned int target = 0 ; 
    gfloat4 ce = mm->getCenterExtent(target);

    LOG(info) << "main drawable ce: " 
              << " x " << ce.x
              << " y " << ce.y
              << " z " << ce.z
              << " w " << ce.w
              ;

    Composition composition ;   
    composition.setDomainCenterExtent(ce);     // index 0 corresponds to entire geometry
    composition.setTimeDomain( gfloat4(0.f, fcfg->getTimeMax(), 0.f, 0.f) );  
    composition.setColorDomain( gfloat4(0.f, numcol, 0.f, 0.f));

    p.add<float>("timeMax",composition.getTimeDomain().y  ); 


    const char* typ ; 
    if(     fcfg->hasOpt("cerenkov"))      typ = "cerenkov" ;
    else if(fcfg->hasOpt("scintillation")) typ = "scintillation" ;
    else                                   typ = "cerenkov" ;

    std::string tag_ = fcfg->getEventTag();
    const char* tag = tag_.empty() ? "1" : tag_.c_str()  ; 

    NPY<float>* npy = NPY<float>::load(typ, tag, det ) ;
    npy->Summary();

    p.add<std::string>("genstepAsLoaded",   npy->getDigestString()  );

    G4StepNPY genstep(npy);    
    genstep.setLookup(lookup); 
   
    if(cache.isJuno())
    {
        LOG(warning) << "main: kludge skip genstep.applyLookup for JUNO " ;
    }
    else
    {   
        genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
    }

    p.add<std::string>("genstepAfterLookup",   npy->getDigestString()  );

    NumpyEvt evt ;
    evt.setOptix(!nooptix);
    evt.setMaxRec(fcfg->getRecordMax());  
    // must set above before setGenStepData to have effect

    evt.setGenstepData(npy); 


    OEngine engine(OEngine::COMPUTE ) ;       
    engine.setFilename(idpath);
    engine.setMergedMesh(mm);   
    engine.setBoundaryLib(blib);   
    engine.setNumpyEvt(&evt);
    engine.setComposition(&composition);                 
    engine.setEnabled(!nooptix);
    engine.setBounceMax(fcfg->getBounceMax());  // 0:prevents any propagation leaving generated photons
    engine.setRecordMax(evt.getMaxRec());       // 1:to minimize without breaking machinery 


    int rng_max = getenvint("CUDAWRAP_RNG_MAX",-1);  // TODO: get rid of envvar
    assert(rng_max >= 1e6); 

    engine.setRngMax(rng_max);


    p.add<std::string>("Type", typ );
    p.add<std::string>("Tag", tag );
    p.add<std::string>("Detector", det );

    p.add<unsigned int>("NumGensteps", evt.getNumGensteps());
    p.add<unsigned int>("RngMax",     engine.getRngMax() );
    p.add<unsigned int>("NumPhotons", evt.getNumPhotons());
    p.add<unsigned int>("NumRecords", evt.getNumRecords());
    p.add<unsigned int>("BounceMax", engine.getBounceMax() );
    p.add<unsigned int>("RecordMax", engine.getRecordMax() );
    p.add<unsigned int>("RepeatIndex", engine.getRecordMax() );


    LOG(info)<< " ******************* main.OptiXEngine::init creating OptiX context, when enabled *********************** " ;
    engine.init();  
    t("initOptiX"); 
    LOG(info) << "main.OptiXEngine::init DONE "; 

   
    LOG(info)<< " ******************* (main) OptiXEngine::generate + propagate  *********************** " ;
    engine.generate();     
    t("generatePropagate"); 

    LOG(info) << "main.OptiXEngine::generate DONE "; 

    engine.downloadEvt();

    NPY<float>* dpho = evt.getPhotonData();
    NPY<short>* drec = evt.getRecordData();
    NPY<unsigned long long>* dhis = evt.getSequenceData();

    t("downloadEvt"); 

    p.add<std::string>("photonData",   dpho->getDigestString()  );
    p.add<std::string>("recordData",   drec->getDigestString()  );
    p.add<std::string>("sequenceData", dhis->getDigestString()  );

    t("checkDigests"); 

    dpho->setVerbose();
    dpho->save("ox%s", typ,  tag, det);

    drec->setVerbose();
    drec->save("rx%s", typ,  tag, det);

    dhis->setVerbose();
    dhis->save("ph%s", typ,  tag, det);

    t("evtSave"); 

    t.stop();

    p.dump();
    t.dump();

    Report r ; 
    r.add(p.getLines()); 
    r.add(t.getLines()); 

    Times* ts = t.getTimes();
    ts->save("$IDPATH/times", Times::name(typ, tag).c_str());

    char rdir[128];
    snprintf(rdir, 128, "$IDPATH/report/%s/%s", tag, typ ); 
    r.save(rdir, Report::name(typ, tag).c_str());  // with timestamp prefix

}



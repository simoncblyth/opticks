FIXED : nljson_causing_cmake_complications
============================================

* Issue was simple ommission of installing FindNLJSON.cmake 


But, I am not using it much if at all...


sysrap/CMakeLists.txt::

  76 find_package(OKConf REQUIRED CONFIG)
  77 find_package(NLJSON REQUIRED MODULE)
  78 find_package(PLog   REQUIRED MODULE)



::


    (ok) A[blyth@localhost sysrap]$ opticks-f SMeta.hh
    ./CSGOptiX/CSGOptiX.cc:#include "SMeta.hh"
    ./sysrap/CMakeLists.txt:   list(APPEND HEADERS  SMeta.hh)
    ./sysrap/SMeta.cc:#include "SMeta.hh"
    ./sysrap/tests/SMetaTest.cc:#include "SMeta.hh"
    (ok) A[blyth@localhost opticks]$ 



Only used here::

    1493 void CSGOptiX::saveMeta(const char* jpg_path) const
    1494 {
    1495     const char* json_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".json");
    1496     LOG(LEVEL) << "[ json_path " << json_path  ;
    1497 
    1498     nlohmann::json& js = meta->js ;
    1499     js["jpg"] = jpg_path ;
    1500     js["emm"] = SGeoConfig::EnabledMergedMesh() ;
    1501 
    1502     if(foundry->hasMeta())
    1503     {
    1504         js["cfmeta"] = foundry->meta ;
    1505     }
    1506 
    1507     std::string extra = SEventConfig::GetGPUMeta();
    1508     js["scontext"] = extra.empty() ? "-" : strdup(extra.c_str()) ;
    1509 
    1510     const std::vector<double>& t = kernel_times ;
    1511     if( t.size() > 0 )
    1512     {
    1513         double mn, mx, av ;
    1514         SVec<double>::MinMaxAvg(t,mn,mx,av);
    1515 
    1516         js["mn"] = mn ;
    1517         js["mx"] = mx ;
    1518         js["av"] = av ;
    1519     }
    1520 
    1521     meta->save(json_path);
    1522     LOG(LEVEL) << "] json_path " << json_path  ;
    1523 }
    1524 



opticks_geospecific_options_in_gdmlaux
=========================================


Add GDMLAux metadata to opticksaux-dx1::

     <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="">
     
    +  <userinfo>
    +     <auxiliary auxtype="opticks_geospecific_options" auxvalue="--boundary MineralOil///Acrylic --pvname /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 "/>
    +  </userinfo> 
    +


::

    epsilon:sysrap blyth$ geocache-create --earlyexit
    ...
    G4GDML: Reading '/usr/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml' done!
    2020-12-17 19:21:41.098 INFO  [3897702] [*CGDML::getUserMeta@310] auxlist 0x7fc984507e38
    2020-12-17 19:21:41.098 INFO  [3897702] [BMeta::dump@199] auxmeta
    {
        "lvmeta": {
            "/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140": {
                "label": "target",
                "lvname": "/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140"
            },
            "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d980x3ee9e20": {
                "SensDet": "SD0",
                "lvname": "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d980x3ee9e20"
            },
            "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400": {
                "SensDet": "SD0",
                "lvname": "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400"
            }
        },
        "usermeta": {
            "opticks_geospecific_options": "--boundary MineralOil///Acrylic --pvname /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 "
        }
    }
    2020-12-17 19:21:41.098 INFO  [3897702] [main@84]  --earlyexit 


::

    epsilon:opticks blyth$ BOpticksResourceTest 
    2020-12-17 20:37:56.541 INFO  [4002341] [BOpticksKey::SetKey@77]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-12-17 20:37:56.543 INFO  [4002341] [BOpticksResource::initViaKey@779] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
                     exename  : OKX4Test
             current_exename  : BOpticksResourceTest
                       class  : X4PhysicalVolume
                     volname  : World0xc15cfc00x40f7000_PV
                      digest  : 50a18baaf29b18fae8c1642927003ee3
                      idname  : OKX4Test_World0xc15cfc00x40f7000_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-12-17 20:37:56.544 INFO  [4002341] [test_getGDMLAuxUserinfo@110] opticks_geospecific_options : [--boundary MineralOil///Acrylic --pvname /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 ]
    2020-12-17 20:37:56.544 INFO  [4002341] [test_getGDMLAuxUserinfoGeospecificOptions@116] --boundary MineralOil///Acrylic --pvname /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 
    epsilon:opticks blyth$ 



Pass the geospecific options from GDMLAux metadata obtained via BOpticksResource into forming 
the embedded opticks commandline. Hence Opticks instanciation can get the detector specific 
options.::

     176 Opticks* G4Opticks::InitOpticks(const char* keyspec, const char* commandline_extra, bool parse_argv ) // static
     177 {
     178     LOG(LEVEL) << "[" ;
     179     LOG(LEVEL) << "[BOpticksResource::Get " << keyspec   ;
     180     BOpticksResource* rsc = BOpticksResource::Get(keyspec) ;
     181     if(keyspec)
     182     {
     183         const char* keyspec2 = rsc->getKeySpec();
     184         assert( strcmp( keyspec, keyspec2) == 0 ); // prior creation of BOpticksResource/BOpticksKey with different spec would trip this
     185     }
     186     LOG(LEVEL) << "]BOpticksResource::Get" ;
     187     LOG(info) << std::endl << rsc->export_();
     188 
     189     const char* geospecific_options = rsc->getGDMLAuxUserinfoGeospecificOptions() ;
     190     LOG(LEVEL) << "GDMLAuxUserinfoGeospecificOptions [" << geospecific_options << "]" ;
     191 
     192     std::string ecl = EmbeddedCommandLine(commandline_extra, geospecific_options) ;
     193     LOG(LEVEL) << "EmbeddedCommandLine : [" << ecl << "]" ;
     194 
     195     LOG(LEVEL) << "[ok" ;
     196     Opticks* ok = NULL ;
     197     if( parse_argv )
     198     {
     199         assert( PLOG::instance && "OPTICKS_LOG is needed to instanciate PLOG" );
     200         const SAr& args = PLOG::instance->args ;
     201         LOG(info) << "instanciate Opticks using commandline captured by OPTICKS_LOG + embedded commandline" ;
     202         args.dump();
     203         ok = new Opticks(args._argc, args._argv, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
     204     }
     205     else
     206     {
     207         LOG(info) << "instanciate Opticks using embedded commandline only " ;
     208         std::cout << ecl << std::endl ;
     209         ok = new Opticks(0,0, ecl.c_str() );  // Opticks instanciation must be after BOpticksKey::SetKey
     210     }







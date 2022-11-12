new_workflow_geometry_configuration
=====================================


OKX4Test is an old workflow executable.  I have moved away from the
old inflexible geocache and OPTICKS_KEY approach. 
To get familiar with the new Opticks
I suggest you start by looking at the new top level G4CXOpticks and tests::

   g4cx/G4CXOpticks.cc
   g4cx/tests/G4CXSimulationTest.cc

You might have seen the below already, in response
to questions on the Opticks mailing list.


For testing geometry loading start from::

  g4cx/tests/G4CXOpticks_setGeometry_Test.sh

The old OPTICKS_KEY approach was too inflexible. It was only really
appropriate when always working with one geometry and always
starting at one level, the persisted GGeo geometry.

In real usage it is necessary to frequently switch between multiple geometries
ranging from single solids to entire geometries and to start from
different levels of the geometry. Examples of levels of geometry:

0. G4VSolid or G4VPhysicalVolume created on the fly : based on some name
1. G4VPhysicalVolume* pointer or path to GDML
2. GGeo pointer or path to persisted GGeo
3. CSGFoundry pointer or path to persisted CSGFoundry

When starting from the lower levels a translation is done to get to the
higher level CSGFoundry geometry that gets uploaded to GPU.
For complex geometries the translation can take a few minutes, 
so it is advantageous to persist the higher level geometry and start from there.
This allows Opticks to load the CSGFoundry binary geometry from file,
upload to GPU and run a small simulation all in under one second even 
with complex geometries such as the JUNO detector.

So, there are now numerous ways to configure which geometry and which
level of geometry to start from using different API. See g4cx/G4CXOpticks.hh::

    55     void setGeometry();
    56     void setGeometry(const char* gdmlpath);
    57     void setGeometry(const G4VPhysicalVolume* world);
    58     void setGeometry(GGeo* gg);
    60     void setGeometry(CSGFoundry* fd);

While you are testing, the argumentless G4CXOpticks::setGeometry() is useful.
The approach it takes is controlled by the existance of envvars::

   135 void G4CXOpticks::setGeometry()
   136 {
   139     if(SSys::hasenvvar(SOpticksResource::OpticksGDMLPath_))
   140     {   
   142         setGeometry(SOpticksResource::OpticksGDMLPath());
   143     }
   144     else if(SSys::hasenvvar(SOpticksResource::SomeGDMLPath_))
   145     {
   147         setGeometry(SOpticksResource::SomeGDMLPath());
   148     }
   149     else if(SSys::hasenvvar(SOpticksResource::CFBASE_))
   150     {
   152         setGeometry(CSGFoundry::Load());
   153     }
   154     else if(SOpticksResource::CFBaseFromGEOM())
   155     {
   159         CSGFoundry* cf = CSGFoundry::Load() ;
   163         setGeometry(cf);
   167     }
   168     else if(SOpticksResource::GDMLPathFromGEOM())
   169     {
   171         setGeometry(SOpticksResource::GDMLPathFromGEOM()) ;
   173     }
   174     else if(SSys::hasenvvar("GEOM"))
   175     {
   177         setGeometry( U4VolumeMaker::PV() );
               // this may load GDML using U4VolumeMaker::PVG if "GEOM"_GDMLPath is defined
   178     }
   179     else
   180     {
   181         LOG(fatal) << " failed to setGeometry " ;
   182         assert(0);
   183     }
   184 }

To avoid always translating you can use the CFBaseFromGEOM approach
which relies on existance of two envvars::

   export GEOM=MyDetectorV1
   export MyDetectorV1_CFBaseFromGEOM=/path/to/directory-that-contains-CSGFoundry-dir

Alternatively to always translate::

   export GEOM=MyDetectorV2
   export MyDetectorV2_GDMLPathFromGEOM=/path/to/geometry.gdml

The GEOM is a short name identifying the geometry
and the associated other envvar that starts with the GEOM value
gives associated information like the path to the CFBase directory or GDML file.
There are other examples in ~/opticks/bin/GEOM_.sh

Note that the name provided by the GEOM envvar is also used
for the automated organization of default output directories
when save methods are used.



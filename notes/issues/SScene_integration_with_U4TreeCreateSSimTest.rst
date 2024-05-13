DONE : SScene_integration_with_U4TreeCreateSSimTest
=======================================================


incorporate SScene creation into U4TreeCreateSSimTest.sh
-----------------------------------------------------------

Initially wanted to incorporate SScene creation into standard U4Tree::Create.
Cannot do that, due to stree (not SSim) arg. But can add to::

   ~/o/u4/tests/U4TreeCreateSSimTest.sh 

So the stree and scene get created together into the SSim folder 


SSim contains scene and stree
-------------------------------

In the normal situation where geometry translation succeeds, the
stree and scene folders are siblings within the SSim NPFold::

    [blyth@localhost SSim]$ pwd
    /home/blyth/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/SSim
 
    [blyth@localhost SSim]$ l
    total 8
    0 -rw-rw-r--. 1 blyth blyth    0 Apr 23 20:30 NPFold_names.txt
    4 -rw-rw-r--. 1 blyth blyth   12 Apr 23 20:30 NPFold_index.txt
    0 drwxr-xr-x. 4 blyth blyth  130 Apr 11 14:02 scene
    0 drwxr-xr-x. 4 blyth blyth   80 Apr 11 14:02 .
    4 drwxr-xr-x. 8 blyth blyth 4096 Mar 28 14:40 stree
    0 drwxr-xr-x. 3 blyth blyth  190 Nov  1  2023 ..

   [blyth@localhost SSim]$ cat NPFold_index.txt 
    stree
    scene


Where the scene is created
---------------------------


::

    241 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    242 {
    243     LOG(LEVEL) << "[ G4VPhysicalVolume world " << world ;
    244     assert(world);
    245     wd = world ;
    246 
    247     assert(sim && "sim instance should have been grabbed/created in ctor" );
    248     stree* st = sim->get_tree();
    249 
    250     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    251     LOG(LEVEL) << "Completed U4Tree::Create " ;
    252 
    253     sim->initSceneFromTree(); // not so easy to do at lower level as do not want to change to SSim arg to U4Tree::Create for headeronly testing   
    254 

    135 SSim::SSim()
    136     :
    137     relp(ssys::getenvvar("SSim__RELP", RELP_DEFAULT )), // alt: "extra/GGeo"
    138     top(nullptr),
    139     extra(nullptr),
    140     tree(new stree),
    141     scene(new SScene)
    142 {
    143     init(); // just sets tree level 
    144 }
    145 


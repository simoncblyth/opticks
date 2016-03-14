Geocache Versioning ?
=======================

Currently trying to load an old geocache into new code asserts::

    simon:optixrap blyth$ ggv --dpib 
    /Users/blyth/env/bin/ggv.sh dumping cmdline arguments
    --dpib
    [2016-03-14 20:09:13.053461] [0x000007fff7057a31] [info]    Opticks::preconfigure argc 2 argv[0] /usr/local/env/graphics/ggeoview/bin/GGeoView mode Interop detector PmtInBox
    App::init OpticksResource::Summary 
    envprefix: OPTICKS_ 
    geokey   : DAE_NAME_DPIB 
    path     : /usr/local/env/geant4/geometry/export/dpib/cfg4.dae 
    query    :  
    ctrl     :  
    digest   : d41d8cd98f00b204e9800998ecf8427e 
    idpath   : /usr/local/env/geant4/geometry/export/dpib/cfg4.d41d8cd98f00b204e9800998ecf8427e.dae 
    meshfix  : (null) 
    [2016-Mar-14 20:09:13.057108]:info: App:: START
    [2016-Mar-14 20:09:13.069403]:info: App::configure NState::description /Users/blyth/.opticks/PmtInBox/State state
    [2016-Mar-14 20:09:13.084819]:info: App:: configure
    [2016-Mar-14 20:09:13.318789]:info: Bookmarks::makeInterpolatedView
    [2016-Mar-14 20:09:13.320322]:info: App:: prepareScene
    [2016-Mar-14 20:09:13.320439]:info: App::loadGeometry START
    [2016-Mar-14 20:09:13.320669]:fatal: NSensorList::read failed to open /usr/local/env/geant4/geometry/export/dpib/cfg4.idmap
    [2016-Mar-14 20:09:13.320827]:info: GGeoLib::loadFromCache ggeo 0x7fd07bc4fa50
    [2016-Mar-14 20:09:13.327982]:info: GMesh::setCenterExtentBuffer  (creates array from buffer)  m_center_extent 0x7fd07bc50240 m_num_solids 6
    [2016-Mar-14 20:09:13.329287]:info: GMesh::updateBounds overwrite solid 0 ce gfloat4      0.000      0.000      0.000    300.000  with gfloat4      0.000      0.000      0.000    300.000 
    [2016-Mar-14 20:09:13.335601]:info: GBndLib::loadIndexBuffer shape 5,4
    [2016-Mar-14 20:09:13.335685]:info: GBndLib::importIndexBuffer BEFORE IMPORT ibuf 5,4 m_bnd.size() 0
    [2016-Mar-14 20:09:13.335784]:info: GBndLib::importIndexBuffer AFTER IMPORT ibuf 5,4 m_bnd.size() 5
    [2016-Mar-14 20:09:13.338914]:info: GMaterialLib::import7,39,4
    Assertion failed: (m_standard_domain->getLength() == nk), function importForTex2d, file /Users/blyth/env/optix/ggeo/GMaterialLib.cc, line 316.
    /Users/blyth/env/graphics/ggeoview/ggeoview.bash: line 1948: 59376 Abort trap: 6           $bin $*
    simon:optixrap blyth$ 


Solution for now is to recreate the geocache by running with "-G".
In future may want to introduce versioning (could be as simple as a version.txt file containing a number).
Then can notice geocache with version incompatible with code and auto recreate the geocache.






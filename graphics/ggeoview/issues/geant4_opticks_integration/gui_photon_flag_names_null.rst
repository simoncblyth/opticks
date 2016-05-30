GUI Photon Flag Names All NULL
================================

progress
---------

* fixed the labels (was an outdated path in GFlags)
* BUT pmt-test GUI selection still not working 


initial issue
---------------

GUI section titled "Photon Flag Sequence Selection" has all the flag names "NULL" 
for both::

   ggv-;ggv-pmt-test --cfg4 --load

   ggv-;ggv-g4gun --dbg --load --target 3153 --optixviz 


Also selecting does not work for pmt-test but it does for g4gun.


But the test seems to get the labels just fine:: 

    simon:ggeoview blyth$ op --gitemindex
    104 -rwxr-xr-x  1 blyth  staff  49200 May 30 19:27 /usr/local/opticks/bin/GItemIndexTest
    OPTICKS_CTRL=volnames
    OPTICKS_MESHFIX=iav,oav
    OPTICKS_GEOKEY=DAE_NAME_DYB
    OPTICKS_BINARY=/usr/local/opticks/bin/GItemIndexTest
    OPTICKS_MESHFIX_CFG=100,100,10,-0.999
    OPTICKS_QUERY=range:3153:12221
    OPTICKS_ARGS=--gitemindex
    OPTICKS_CMD=--gitemindex
    proceeding : /usr/local/opticks/bin/GItemIndexTest --gitemindex
    [2016-05-30 19:28:57.190406] [0x000007fff74d6331] [info]    Types::readFlags path[$ENV_HOME/optickscore/OpticksPhoton.h]
    [2016-05-30 19:28:57.191390] [0x000007fff74d6331] [info]    Types::readFlags pair count 15
    /usr/local/opticks/bin/GItemIndexTest ixdir /usr/local/env/opticks/PmtInBox/ixtorch
    [2016-05-30 19:28:57.192117] [0x000007fff74d6331] [debug]   loadMap /usr/local/env/opticks/PmtInBox/ixtorch/-4/History_SequenceSource.json
    [2016-05-30 19:28:57.192144] [0x000007fff74d6331] [debug]   jsonutil.loadTree:  load path: /usr/local/env/opticks/PmtInBox/ixtorch/-4/History_SequenceSource.json
    [2016-05-30 19:28:57.193064] [0x000007fff74d6331] [debug]   loadMap /usr/local/env/opticks/PmtInBox/ixtorch/-4/History_SequenceLocal.json
    [2016-05-30 19:28:57.193076] [0x000007fff74d6331] [debug]   jsonutil.loadTree:  load path: /usr/local/env/opticks/PmtInBox/ixtorch/-4/History_SequenceLocal.json
    [2016-05-30 19:28:57.194353] [0x000007fff74d6331] [info]    GItemIndex::gui_radio_select_debug names 32 labels 32
     title History_Sequence
      0 name                   8cd label                                   8cd TORCH BT SA 
      1 name                   7cd label                                   7cd TORCH BT SD 
      2 name                  8ccd label                               8ccd TORCH BT BT SA 
      3 name                    4d label                                       4d TORCH AB 
      4 name                   86d label                                   86d TORCH BS SA 
      5 name                   4cd label                                   4cd TORCH BT AB 
      6 name                  4ccd label                               4ccd TORCH BT BT AB 
      7 name                   8bd label                                   8bd TORCH BR SA 
      8 name                  8c6d label                               8c6d TORCH BS BT SA 
      9 name                 86ccd label                           86ccd TORCH BT BT BS SA 
     10 name                   46d label                                   46d TORCH BS AB 
     11 name                8cbbcd label                       8cbbcd TORCH BT BR BR BT SA 
     12 name                   4bd label                                   4bd TORCH BR AB 
     13 name                  7c6d label                               7c6d TORCH BS BT SD 
     14 name                  866d label                               866d TORCH BS BS SA 
     15 name                 8cc6d label                           8cc6d TORCH BS BT BT SA 




The ImGui displayed text comes from::

    oglrap-/Photons
    ggeo-/GItemIndex 
    npy-/Index 


::

    [2016-May-30 18:58:57.082774]:info: App:: uploadEvtViz
    [2016-May-30 18:58:57.082888]:info: Bookmarks::create : persisting state to slot 0
    [2016-May-30 18:58:57.082981]:info: Bookmarks::collect 0
    [2016-May-30 18:58:57.083679]:warning: Caught bad lexical cast with error bad lexical cast: source type value could not be interpreted as target
    [2016-May-30 18:58:57.084081]:info: Types::readFlags path[$ENV_HOME/optickscore/OpticksPhoton.h]
    [2016-May-30 18:58:57.084576]:info: Types::readFlags pair count 15
    [2016-May-30 18:58:57.085133]:info: Index::save sname GFlagIndexSource.ini lname GFlagIndexLocal.ini itemtype GFlagIndex ext .ini
    [2016-May-30 18:58:57.086167]:info: enter runloop 

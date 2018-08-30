OpticksResourceTest_j1808_geokey_not_getting_thru_when_have_no_dae
===================================================================

GDML from 10.4.2 may be sufficient without the help of a G4DAE export, 
trying to get that to work with j1808 geometry.

But find the geokey is not getting thru.

* probably a .dae existance check and fallback to DYB in OpticksResource or BOpticksResource ?

  * NOPE : the opticksdata.ini had unexpected OPTICKS_GEOKEY for unknown reasons 

* where is the opticksdata ini file read, what comes from from ini and what from env ?

::

    epsilon:export blyth$ VERBOSE=1 op --resource --j1808 --dumpenv
    === op-cmdline-binary-match : finds 1st argument with associated binary : --resource
    op-cmdline-parse OPTICKS_GEO : J1808
    op-geometry-setup-juno : geo J1808
    op-cmdline-parse : OPTICKS_GEOKEY OPTICKSDATA_DAEPATH_J1808
    264 -rwxr-xr-x  1 blyth  staff  132112 Aug 29 22:28 /usr/local/opticks/lib/OpticksResourceTest
    proceeding.. : /usr/local/opticks/lib/OpticksResourceTest --resource --j1808 --dumpenv
    MAIN OPTICKS_GEOKEY : OPTICKSDATA_DAEPATH_J1808
    /usr/local/opticks/lib/OpticksResourceTest --resource --j1808 --dumpenv
    PLOG::PLOG  instance 0x7fdd54403540 this 0x7fdd54403540 logpath /usr/local/opticks/lib/OpticksResourceTest.log
    2018-08-30 10:46:55.066 INFO  [3900851] [BOpticksKey::SetKey@42] BOpticksKey::SetKey from OPTICKS_KEY envvar (null)
    2018-08-30 10:46:55.066 INFO  [3900851] [SLog::SLog@12] Opticks::Opticks 
    2018-08-30 10:46:55.067 ERROR [3900851] [BOpticksResource::init@81] layout : 1
    OPTICKS_INSTALL_PREFIX=/usr/local/opticks
    OPTICKS_CTRL=
    OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_DYB



Huh whats OPTICKS_GEOKEY doing in there::

    epsilon:boostrap blyth$ cat /usr/local/opticks/opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DFAR=/usr/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_DLIN=/usr/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_DPIB=/usr/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_J1707=/usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    OPTICKSDATA_DAEPATH_J1808=/usr/local/opticks/opticksdata/export/juno1808/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/usr/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LXE=/usr/local/opticks/opticksdata/export/LXe/g4_00.dae
    OPTICKS_GEOKEY=OPTICKSDATA_DAEPATH_DYB
    epsilon:boostrap blyth$ 


::

    232 opticksdata-export-ini()
    233 {
    234    local msg="=== $FUNCNAME :"
    235 
    236    opticksdata-export
    237 
    238    local ini=$(opticksdata-ini)
    239    local dir=$(dirname $ini)
    240    mkdir -p $dir
    241 
    242    echo $msg writing OPTICKS_DAEPATH_ environment to $ini
    243    env | grep OPTICKSDATA_DAEPATH_ | sort > $ini
    244 
    245    cat $ini
    246 }


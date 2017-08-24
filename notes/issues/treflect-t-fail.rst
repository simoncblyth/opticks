treflect-t fail
==================

::

    simon:opticksnpy blyth$ treflect-
    simon:opticksnpy blyth$ treflect-t
    treflect-- is a function
    treflect-- () 
    { 
        type $FUNCNAME;
        local cmdline=$*;
        local pol;
        if [ "${cmdline/--spol}" != "${cmdline}" ]; then
            pol=s;
            cmdline=${cmdline/--spol};
        else
            if [ "${cmdline/--ppol}" != "${cmdline}" ]; then
                pol=p;
                cmdline=${cmdline/--ppol};
            else
                pol=s;
            fi;
        fi;
        case $pol in 
            s)
                tag=$(treflect-stag)
            ;;
            p)
                tag=$(treflect-ptag)
            ;;
        esac;
        if [ "${cmdline/--tcfg4}" != "${cmdline}" ]; then
            tag=-$tag;
        fi;
        echo pol $pol tag $tag;
        local photons=1000000;
        local material=MainH2OHale;
        local torch_config=(type=refltest photons=$photons mode=${pol}pol,flatTheta polarization=0,0,-1 frame=-1 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000 source=0,0,-200 radius=100 distance=25 zenithazimuth=0.5,1,0,1 material=Vacuum wavelength=550);
        local test_config=(mode=BoxInBox analytic=1 shape=box parameters=0,0,0,1000 boundary=Rock//perfectAbsorbSurface/Vacuum shape=box parameters=0,0,0,200 boundary=Vacuum///$material);
        op.sh $* --animtimemax 7 --timemax 7 --geocenter --eye 0,0,1 --test --testconfig "$(join _ ${test_config[@]})" --torch --torchconfig "$(join _ ${torch_config[@]})" --torchdbg --save --tag $tag --cat $(treflect-det)
    }
    pol s tag 1
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --spol --compute --animtimemax 7 --timemax 7 --geocenter --eye 0,0,1 --test --testconfig mode=BoxInBox_analytic=1_shape=box_parameters=0,0,0,1000_boundary=Rock//perfectAbsorbSurface/Vacuum_shape=box_parameters=0,0,0,200_boundary=Vacuum///MainH2OHale --torch --torchconfig type=refltest_photons=1000000_mode=spol,flatTheta_polarization=0,0,-1_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,-200_radius=100_distance=25_zenithazimuth=0.5,1,0,1_material=Vacuum_wavelength=550 --torchdbg --save --tag 1 --cat reflect
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OKTest
    288 -rwxr-xr-x  1 blyth  staff  143804 Aug 23 12:01 /usr/local/opticks/lib/OKTest
    proceeding.. : /usr/local/opticks/lib/OKTest --spol --compute --animtimemax 7 --timemax 7 --geocenter --eye 0,0,1 --test --testconfig mode=BoxInBox_analytic=1_shape=box_parameters=0,0,0,1000_boundary=Rock//perfectAbsorbSurface/Vacuum_shape=box_parameters=0,0,0,200_boundary=Vacuum///MainH2OHale --torch --torchconfig type=refltest_photons=1000000_mode=spol,flatTheta_polarization=0,0,-1_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,-200_radius=100_distance=25_zenithazimuth=0.5,1,0,1_material=Vacuum_wavelength=550 --torchdbg --save --tag 1 --cat reflect
    2017-08-24 15:34:50.664 INFO  [354788] [OpticksQuery::dump@79] OpticksQuery::init queryType range query_string range:3153:12221 query_name NULL query_index 0 query_depth 0 no_selection 0 nrange 2 : 3153 : 12221
    2017-08-24 15:34:50.665 INFO  [354788] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest 96ff965744a2f6b78c24e33c80d3a4cd age.tot_seconds 4478975 age.tot_minutes 74649.586 age.tot_hours 1244.160 age.tot_days     51.840
    2017-08-24 15:34:50.839 INFO  [354788] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-24 15:34:50.970 INFO  [354788] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-24 15:34:51.061 INFO  [354788] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-08-24 15:34:51.209 INFO  [354788] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-24 15:34:51.209 INFO  [354788] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-08-24 15:34:51.209 INFO  [354788] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-08-24 15:34:51.209 INFO  [354788] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-08-24 15:34:51.232 INFO  [354788] [GGeo::loadAnalyticPmt@764] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-08-24 15:34:51.245 WARN  [354788] [GGeoTestConfig::getArg@171] GGeoTestConfig::getArg UNRECOGNIZED arg shape
    2017-08-24 15:34:51.245 WARN  [354788] [GGeoTestConfig::set@194] GGeoTestConfig::set WARNING ignoring unrecognized parameter box
    2017-08-24 15:34:51.245 WARN  [354788] [GGeoTestConfig::getArg@171] GGeoTestConfig::getArg UNRECOGNIZED arg shape
    2017-08-24 15:34:51.246 WARN  [354788] [GGeoTestConfig::set@194] GGeoTestConfig::set WARNING ignoring unrecognized parameter box
    2017-08-24 15:34:51.246 WARN  [354788] [GGeoTest::init@54] GGeoTest::init booting from m_ggeo 
    2017-08-24 15:34:51.246 WARN  [354788] [GMaker::init@176] GMaker::init booting from cache
    2017-08-24 15:34:51.246 INFO  [354788] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-24 15:34:51.379 INFO  [354788] [*GMergedMesh::load@631] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-24 15:34:51.384 INFO  [354788] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-24 15:34:51.384 INFO  [354788] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-08-24 15:34:51.384 INFO  [354788] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-08-24 15:34:51.384 INFO  [354788] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-08-24 15:34:51.387 FATAL [354788] [GGeoTestConfig::getNumElements@210] GGeoTestConfig::getNumElements ELEMENT MISMATCH IN TEST GEOMETRY CONFIGURATION  nbnd (boundaries) 2 nnod (nodes) 0 npar (parameters) 2 ntra (transforms) 0
    Assertion failed: (equal && "need equal number of boundaries, parameters, transforms and nodes"), function getNumElements, file /Users/blyth/opticks/ggeo/GGeoTestConfig.cc, line 218.
    /Users/blyth/opticks/bin/op.sh: line 689: 67949 Abort trap: 6           /usr/local/opticks/lib/OKTest --spol --compute --animtimemax 7 --timemax 7 --geocenter --eye 0,0,1 --test --testconfig mode=BoxInBox_analytic=1_shape=box_parameters=0,0,0,1000_boundary=Rock//perfectAbsorbSurface/Vacuum_shape=box_parameters=0,0,0,200_boundary=Vacuum///MainH2OHale --torch --torchconfig type=refltest_photons=1000000_mode=spol,flatTheta_polarization=0,0,-1_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,-200_radius=100_distance=25_zenithazimuth=0.5,1,0,1_material=Vacuum_wavelength=550 --torchdbg --save --tag 1 --cat reflect
    /Users/blyth/opticks/bin/op.sh RC 134




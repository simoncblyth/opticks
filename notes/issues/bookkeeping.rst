bookeeping thoughts
=======================

idea for higher level metadata refering to a group of runs
------------------------------------------------------------

When launching a group of runs with eg::

    geocache-bench --xanalytic --enabledmergedmesh 2

It results in multiple invokations of OpSnapTest each writing into its
own results directory. This metadata is parsed and presented by bench.py::


    [blyth@localhost opticks]$ bench.py --digest 52 --since 6pm
    Namespace(digest='52', exclude=None, include=None, metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='6pm')
    since : 2019-05-23 18:00:00 

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558620718 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_221158  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.037      1.000      0.342           1.580    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_221158  
                          R1_TITAN_V      0.045      1.214      0.416           1.501    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_221158  
            R0_TITAN_V_AND_TITAN_RTX      0.058      1.550      0.531           2.408    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_221158  
                          R0_TITAN_V      0.090      2.425      0.830           1.511    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_221158  
                        R0_TITAN_RTX      0.109      2.920      1.000           1.400    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_221158  

    ---
     OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig steps=5,eyestartz=-1,eyestopz=-0.5 --size 5120,2880,1 --embedded --cvd 1 --rtx 1 --runfolder geocache-bench --runstamp 1558621167 --runlabel R1_TITAN_RTX --xanalytic --enabledmergedmesh 2
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.52e273e4ad5423fe2fc8aa44bbf055ec
    /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/52e273e4ad5423fe2fc8aa44bbf055ec/1
                    20190523_221927  launchAVG      rfast      rslow      prelaunch000 
                        R1_TITAN_RTX      0.030      1.000      0.280           1.548    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_RTX/20190523_221927  
                          R1_TITAN_V      0.041      1.358      0.380           1.500    : /tmp/blyth/location/results/geocache-bench/R1_TITAN_V/20190523_221927  
            R0_TITAN_V_AND_TITAN_RTX      0.058      1.917      0.537           1.717    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V_AND_TITAN_RTX/20190523_221927  
                          R0_TITAN_V      0.091      3.005      0.841           1.093    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_V/20190523_221927  
                        R0_TITAN_RTX      0.109      3.573      1.000           0.996    : /tmp/blyth/location/results/geocache-bench/R0_TITAN_RTX/20190523_221927  
    Namespace(digest='52', exclude=None, include=None, metric='launchAVG', name='geocache-bench', other='prelaunch000', resultsdir='$TMP/results', since='6pm')
    [blyth@localhost opticks]$ 



Thats fine, but it would be good to record higher level metadata for the each group of runs, 
such as the high level command line::
  
    geocache-bench --xanalytic --enabledmergedmesh 2

As well as a comment on what changed eg "remove the hemi ellipsoid bug", that can be parsed and reported by bench.py
so the list of groups of runs is easier to follow. 


how to implement
------------------

* perhaps write to a single folder using a python script ? could swallow some arguments and write json ?

  * /tmp/blyth/location/results/geocache-bench/GROUPMETA/20190523_221927

* hmm maybe simpler to just duplicate the groupmeta.json into all the dated folders and do it all at C++ level
  for uniformity 




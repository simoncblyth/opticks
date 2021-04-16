OpSnapTest_STimes_perplexing
================================




2019
-------

With RTX OFF the prelaunch is taking ages.

Hmm the presentation is misleading

1. validate/compile/prelaunch only happens once, with only the launch times changing 
   but the averages of validate/compile/prelaunch go down as the count goes up

2. the compile time is always zero : looks like with OptiX 600 the actual compilation is done in the prelaunch


::

    [blyth@localhost opticks]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0 OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 
    2019-04-21 00:27:07.131 INFO  [107202] [BOpticksKey::SetKey@45] BOpticksKey::SetKey from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    ...
    2019-04-21 00:27:12.438 FATAL [107202] [OpticksAim::setTarget@121] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 352851 aim 1 ce 0.0000,0.0000,19785.0000,1965.0000
    2019-04-21 00:27:12.438 INFO  [107202] [Composition::setCenterExtent@1180] Composition::setCenterExtent ce 0.0000,0.0000,19785.0000,1965.0000
    2019-04-21 00:27:12.515 INFO  [107202] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-661.684) front 0.6061,0.6061,-0.5152
    2019-04-21 00:27:19.175 INFO  [107202] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     1 
     validate      0.0546     0.0546 
     compile       0.0000     0.0000 
     prelaunch     6.5256     6.5256 
     launch        0.0155     0.0155 

    2019-04-21 00:27:19.175 INFO  [107202] [OContext::snap@681]  path /tmp/snap00000.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00000.ppm
     i     1 eyez   0.838889 path /tmp/snap00001.ppm
    2019-04-21 00:27:19.225 INFO  [107202] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     2 
     validate      0.0546     0.0273 
     compile       0.0000     0.0000 
     prelaunch     6.5256     3.2628 
     launch        0.0418     0.0209 

    2019-04-21 00:27:19.225 INFO  [107202] [OContext::snap@681]  path /tmp/snap00001.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00001.ppm
     i     2 eyez   0.827778 path /tmp/snap00002.ppm
    2019-04-21 00:27:19.270 INFO  [107202] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     3 
     validate      0.0546     0.0182 
     compile       0.0000     0.0000 
     prelaunch     6.5256     2.1752 
     launch        0.0655     0.0218 

    2019-04-21 00:27:19.270 INFO  [107202] [OContext::snap@681]  path /tmp/snap00002.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00002.ppm
     i     3 eyez   0.816667 path /tmp/snap00003.ppm
    2019-04-21 00:27:19.315 INFO  [107202] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     4 
     validate      0.0546     0.0137 
     compile       0.0000     0.0000 
     prelaunch     6.5256     1.6314 
     launch        0.0892     0.0223 

    2019-04-21 00:27:19.315 INFO  [107202] [OContext::snap@681]  path /tmp/snap00003.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00003.ppm
     i     4 eyez   0.805556 path /tmp/snap00004.ppm
    2019-04-21 00:27:19.357 INFO  [107202] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     5 
     validate      0.0546     0.0109 
     compile       0.0000     0.0000 
     prelaunch     6.5256     1.3051 
     launch        0.1105     0.0221 


Compared to RTX ON::

    [blyth@localhost opticks]$ CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=1 OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 
    2019-04-21 00:29:16.650 INFO  [110589] [BOpticksKey::SetKey@45] BOpticksKey::SetKey from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    ...
    2019-04-21 00:29:22.009 FATAL [110589] [OpticksAim::setTarget@121] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 352851 aim 1 ce 0.0000,0.0000,19785.0000,1965.0000
    2019-04-21 00:29:22.009 INFO  [110589] [Composition::setCenterExtent@1180] Composition::setCenterExtent ce 0.0000,0.0000,19785.0000,1965.0000
    2019-04-21 00:29:22.089 INFO  [110589] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-661.684) front 0.6061,0.6061,-0.5152
    2019-04-21 00:29:23.398 INFO  [110589] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     1 
     validate      0.0534     0.0534 
     compile       0.0000     0.0000 
     prelaunch     1.1415     1.1415 
     launch        0.0505     0.0505 

    2019-04-21 00:29:23.400 INFO  [110589] [OContext::snap@681]  path /tmp/snap00000.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00000.ppm
     i     1 eyez   0.838889 path /tmp/snap00001.ppm
    2019-04-21 00:29:23.483 INFO  [110589] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     2 
     validate      0.0534     0.0267 
     compile       0.0000     0.0000 
     prelaunch     1.1415     0.5708 
     launch        0.1096     0.0548 

    2019-04-21 00:29:23.483 INFO  [110589] [OContext::snap@681]  path /tmp/snap00001.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00001.ppm
     i     2 eyez   0.827778 path /tmp/snap00002.ppm
    2019-04-21 00:29:23.559 INFO  [110589] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     3 
     validate      0.0534     0.0178 
     compile       0.0000     0.0000 
     prelaunch     1.1415     0.3805 
     launch        0.1649     0.0550 

    2019-04-21 00:29:23.559 INFO  [110589] [OContext::snap@681]  path /tmp/snap00002.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00002.ppm
     i     3 eyez   0.816667 path /tmp/snap00003.ppm
    2019-04-21 00:29:23.635 INFO  [110589] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     4 
     validate      0.0534     0.0134 
     compile       0.0000     0.0000 
     prelaunch     1.1415     0.2854 
     launch        0.2205     0.0551 

    2019-04-21 00:29:23.635 INFO  [110589] [OContext::snap@681]  path /tmp/snap00003.ppm width 1920 width 1920 height 1080 height 1080 depth 1
    Wrote file /tmp/snap00003.ppm
     i     4 eyez   0.805556 path /tmp/snap00004.ppm
    2019-04-21 00:29:23.711 INFO  [110589] [OTracer::trace_@150] OTracer::trace m_trace_times 
     count     5 
     validate      0.0534     0.0107 
     compile       0.0000     0.0000 
     prelaunch     1.1415     0.2283 
     launch        0.2753     0.0551 






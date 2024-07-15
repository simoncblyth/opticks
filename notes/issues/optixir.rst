optixir
=========

* https://forums.developer.nvidia.com/t/embedding-optix-ir/273199
* https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/MDL_renderer/src/Device.cpp#L721



crovella blog
--------------

* https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/

* https://developer.nvidia.com/blog/author/bob-crovella/


nvidia optix profile nsight
-----------------------------

* https://forums.developer.nvidia.com/t/need-help-profiling-an-optix-application/265266/2


droettger::

    I would expect this to work with PTX. I normally do not compile my OptiX device
    code with any debug flags (-g, -G) though because that doesn’t make sense for
    profiling anyway.  Having the -lineinfo should be enough to get the source code
    connections. The line infos should appear interleaved inside the PTX code and
    the CUDA source files are listed at the bottom of the PTX file.



* https://forums.developer.nvidia.com/t/nsight-compute-command-line-profiler-for-optix-kernels/296747



profile optix : highlighted informative posts
----------------------------------------------

* https://forums.developer.nvidia.com/t/profiling-optix/231192/4



profile optix
----------------

* https://forums.developer.nvidia.com/search?q=profile%20optix%20

* https://forums.developer.nvidia.com/t/optix-profiling/270780/7

* https://forums.developer.nvidia.com/t/optix-profiling-using-nsight/70610

* https://on-demand.gputechconf.com/siggraph/2018/video/sig1843-johann-korndoerfer-nsight-compute.html



* https://forums.developer.nvidia.com/search?q=profile%20optix%20%20order%3Alatest



Acceleration Structure Viewer
-------------------------------

* https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#as-viewer



Nsight Compute
---------------

Lots of intro vids

* https://developer.nvidia.com/nsight-compute
* https://developer.nvidia.com/nsight-compute


NVIDIA Nsight™ Compute is an interactive profiler for CUDA® and NVIDIA OptiX™
that provides detailed performance metrics and API debugging via a user
interface and command-line tool. Users can run guided analysis and compare
results with a customizable and data-driven user interface, as well as
post-process and analyze results in their own workflows.



NVIDIA Nsight Compute is an interactive kernel profiler for CUDA applications

* https://docs.nvidia.com/nsight-compute/NsightCompute/index.html


* https://forums.developer.nvidia.com/t/nvidia-nsight-compute-2024-1-is-now-available/279800


included with CUDA
--------------------

::

   /usr/local/cuda-11.7/nsight-compute-2022.2.0


docs
------

::

    /usr/local/cuda-11.7/nsight-compute-2022.2.0/docs/index.html


https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html#library-support

The interactive profile activity is only supported for OptiX applications if
the OPTIX_FORCE_DEPRECATED_LAUNCHER variable is set to CBL1 in the
application’s environment.


example
----------

https://indico.cern.ch/event/962112/contributions/4110591/attachments/2159863/3643851/CERN_Nsight_Compute.pdf

::

    By default, CLI results are printed to stdout
    Use --export/-o to save results to a report file, use -f to force overwrite
    $ ncu -f -o $HOME/my_report <app>
    $ my_report.ncu-rep
    Use --log-file to pipe text output to a different stream (stdout/stderr/file)
    Can use (env) variables available in your batch script or file macros to add report name placeholders
    Full parity with nvprof filename placeholders/file macros
    $ ncu -f -o $HOME/my_report_%h_${LSB_JOBID}_%p <app>
    $ my_report_host01_951697_123.ncu-rep



ncu binary
------------

* https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html


::

    P[blyth@localhost nsight-compute-2022.2.0]$ /usr/local/cuda-11.7/nsight-compute-2022.2.0/ncu --help
    usage: ncu [options] [program] [program-arguments]

    General Options:
      -h [ --help ]                         Print this help message.
      -v [ --version ]                      Print the version number.
      --mode arg (=launch-and-attach)       Select the mode of interaction with the target application:
                                              launch-and-attach
                                              (launch and attach for profiling)
                                              launch
                                              (launch and suspend for later attach)
                                              attach
                                              (attach to launched application)
      -p [ --port ] arg (=49152)            Base port for connecting to target application
      --max-connections arg (=64)           Maximum number of ports for connecting to target application

    Launch Options:
      --check-exit-code arg (=1)            Check the application exit code and print an error if it is different than 0. 
                                            If set, --replay-mode application will stop after the first pass if the exit 
                                            code is not 0.

::

    P[blyth@localhost ~]$ /usr/local/cuda-11.7/nsight-compute-2022.2.0/ncu --list-chips   
    ga100, ga102, ga103, ga104, ga106, ga107, ga10b, gv100, gv11b, tu102, tu104, tu106, tu116, tu117
    P[blyth@localhost ~]$ 


::

    P[blyth@localhost ~]$ /usr/local/cuda-11.7/nsight-compute-2022.2.0/ncu --list-sets
    ---------- --------------------------------------------------------------------------- ------- -----------------
    Identifier Sections                                                                    Enabled Estimated Metrics
    ---------- --------------------------------------------------------------------------- ------- -----------------
    default    LaunchStats, Occupancy, SpeedOfLight                                        yes     36               
    detailed   ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnaly no      181              
               sis, Occupancy, SchedulerStats, SourceCounters, SpeedOfLight, SpeedOfLight_                          
               RooflineChart, WarpStateStats                                                                        
    full       ComputeWorkloadAnalysis, InstructionStats, LaunchStats, MemoryWorkloadAnaly no      198              
               sis, MemoryWorkloadAnalysis_Chart, MemoryWorkloadAnalysis_Tables, Nvlink_Ta                          
               bles, Nvlink_Topology, Occupancy, SchedulerStats, SourceCounters, SpeedOfLi                          
               ght, SpeedOfLight_RooflineChart, WarpStateStats                                                      
    roofline   SpeedOfLight, SpeedOfLight_HierarchicalDoubleRooflineChart, SpeedOfLight_Hi no      48               
               erarchicalHalfRooflineChart, SpeedOfLight_HierarchicalSingleRooflineChart,                           
               SpeedOfLight_HierarchicalTensorRooflineChart, SpeedOfLight_RooflineChart                             
    source     SourceCounters                                                              no      67               
    P[blyth@localhost ~]$ 




ProfilingGuide
----------------

* https://docs.nvidia.com/search/index.html?page=1&sort=relevance&term=nsight%20compute%20Profiling
* https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf
* ~/opticks_refs/Nsight_Compute_ProfilingGuide.pdf




Recommendations for splitting work between GPUs
-------------------------------------------------

* https://forums.developer.nvidia.com/t/recommendations-for-splitting-work-between-gpus/282405



overview
----------


* https://gpuhackshef.readthedocs.io/en/latest/tools/nvidia-profiling-tools.html


nsight compute optix
----------------------


* https://on-demand.gputechconf.com/siggraph/2019/pdf/sig915-optix-performance-tools-tricks.pdf
* ~/opticks_refs/sig915-optix-performance-tools-tricks.pdf


* https://sourcesup.renater.fr/wiki/atelieromp/_media/cuda_developertools_nsightoverview_fcourteille_28juillet2020.pdf

VecGeom GPU (25th Geant4 Collaboration Meeting, VecGeom@GPU, andrei.gheata@cern.ch)
------------------------------------------------------------------------------------

* https://indico.cern.ch/event/942142/contributions/4016084/attachments/2102467/3535531/VecGeomGPU_ongoing_work.pdf





Perlmutter
-------------

* https://docs.nersc.gov/performance/readiness/

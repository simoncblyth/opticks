geometry_scanning_to_check_for_bottlenecks
==============================================

How scanning works
-------------------

The pseudo-dynamic geometry capabilities of CSGFoundry::Load are used to 
load the persisted geometry from cache and then selectively copy 
the geometry to a separate CSGFoundry with selection configured by 
envvars ELV and EMM that is then uploaded when creating the CSGOptiX 
geometry. 

* ELV envvar controls  at the level of LV (logical volume) shapes.
* EMM envvar controls at the coarser level of compound solids 

When scanning each invokation of the rendering executable typically 
takes around 2 seconds, as the full geometry is loaded from file 
and then selectively copied and uploaded for each render. 

A faster approach would be to implement real dynamic geometry changing, 
as opposed to the pseudo-dynamic current approach that relies on 
repeatedly reloading the geometry from file. Real dynamic geometry 
would be able to read from file once only and then clear the GPU geometry 
and remake it from within the same process. Although faster 
scanning would be useful it is not currentlt a priority.  


Scanning Scripts
------------------

Create renders varying EMM and ELV envvars, changing geometry::

    ~/o/CSGOptiX/cxr_scan_emm.sh    ## symbolic link to the cxr_scan.sh script
    ~/o/CSGOptiX/cxr_scan_elv.sh    ## symbolic link to the cxr_scan.sh script

These commands are typically repeated several times, varying which set of 
envvar controlled scans are used for each pass. For example excluding geometry
shapes one-by-one in one pass and including shapes one-by-one it in a second 
as well as doing some "candle" reference renders such as everything.  


cxr_view.sh vs cxr_overview.sh
----------------------------------

The machinery of the scanning and analysis is kept independent 
of the details of the viewpoint to render.

cxr_view.sh
    view of a specific volume from within the geometry   

cxr_overview.sh 
    distant view of the full geometry 

In some cases some irregularity in render times has been observed
due to variations in view distances resulting from the changing sizes of the 
axis aligned bbox of the guidetube.  


Analysis/presentation of render metadata using elv.sh and emm.sh
------------------------------------------------------------------

elv.sh::

    ~/o/CSGOptiX/elv.sh txt ## write TXT table listing ordered render times with geometry variations                                                                 
    ~/o/CSGOptiX/elv.sh rst ## write RST formatted table with same info as above
    ~/o/CSGOptiX/elv.sh jpg ## write list of jpg paths in ascending render time order         

Similarly with emm.sh::

    ~/o/CSGOptiX/emm.sh txt ## write TXT table listing ordered render times with geometry variations                                                                 
    ~/o/CSGOptiX/emm.sh rst ## write RST formatted table with same info as above
    ~/o/CSGOptiX/emm.sh jpg ## write list of jpg paths in ascending render time order         


The elv.sh and emm.sh are actually via symbolic link the same script that 
sets operation mode based on the script BASH_SOURCE. 

The script sets up the SNAP envvars and runs the "jstab" sub-command of either the cxr_view.sh 
or cxr_overview.sh script which sets further environment and then runs the snap.py python script
which finds and loads the render metadata json and presents it as configured. 

  
How to spot problems
----------------------

Render times depend on how well the acceleration structure built by OptiX (eg using a BVH structure) 
can be traversed to find intersects. When a geometry has problem then typically certain shapes will 
have a large impact on overall intersect performance. Identify the causes of such problems by varying 
what is included or excluded within the geometry and analysing rendering times.   

* problem shapes will often stick out with very long render times when included or speedups when excluded
  (very large effects have been observed with slowdowns of factors of hundreds)

When the variation across the range of different geometries is small, then it is 
possible to conclude that no large geometry issues remain. 





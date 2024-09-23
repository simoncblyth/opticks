geometry_scanning_to_check_for_bottlenecks
==============================================

How scanning works
-------------------

The dynamic geometry capabilities of CSGFoundry::Load are used to 
load the persisted geometry from cache and then selectively copy 
the geometry to a separate CSGFoundry with selection configured by 
envvars ELV and EMM that is then uploaded when creating the CSGOptiX 
geometry. 

* ELV controls at the level of LV (logical volume) shapes.
* EMM controls at the coarser level of compound solids 


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

  


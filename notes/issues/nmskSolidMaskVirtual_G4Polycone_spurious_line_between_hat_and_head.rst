nmskSolidMaskVirtual_G4Polycone_spurious_line_between_hat_and_head
=====================================================================

Related
---------

* prev :doc:`ct_scan_nmskTailInner`

setup for ct scan
--------------------

::

    gc
    ./mtranslate.sh   # after adding nmskSolidMaskVirtual__U1 to the geomlist 

    c
    ./ct.sh   ## CSGSimtraceTest


issue
--------

* line of spurious : between hat and head : but note also one spurious not on that line 

* zooming in the ct scan corner  however shows that standard coincidence avoidance 
  actually can be used because the lower hat radius is less than the head cylinder radius 




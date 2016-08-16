Offset Vacuum Base
=======================

This issue prompted a review of geometry handling in :doc:`/graphics/ggeoview/geometry_review`


.. image:: //env/graphics/ggeoview/issues/offset_bottom/dpib.png
   :width: 900px
   :align: center


Live created PmtInBox features the offset bottom very clearly

::

    ggv-;ggv-pmt-test --tracer


.. image:: //env/graphics/ggeoview/issues/offset_bottom/pmttest-slice-offset.png
   :width: 900px
   :align: center


G4DAE export with cfg4-dpib loaded back in via geocache DOES NOT SHOW THE OFFSET ?::

    ggv --dpib --tracer


.. image:: //env/graphics/ggeoview/issues/offset_bottom/dpib-sliced-not-offset.png
   :width: 900px
   :align: center



Plotting mesh vertices together with analytic shapes.



Standard geocache exhibits the issue:

.. image:: //env/graphics/ggeoview/issues/offset_bottom/analytic_vs_triangulated_standard_vacuum_offset.png
   :width: 900px
   :align: center


Using a recent G4DAE(g4d-) export doesnt have the issue, here DPIB_ALL:

.. image:: //env/graphics/ggeoview/issues/offset_bottom/analytic_vs_triangulated_dpib_all.png
   :width: 900px
   :align: center



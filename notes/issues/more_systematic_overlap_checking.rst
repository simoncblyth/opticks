more_systematic_overlap_checking
=================================



Old simtrace approach is manual and python based comparing two pointclouds
----------------------------------------------------------------------------

* see cx cxt_min.py


Comparing bbox with stree_load_test.sh works but sibling bbox overlap very commonly is not a problem
------------------------------------------------------------------------------------------------------

::

    TEST=get_global_aabb_sibling_overlaps ~/o/sysrap/tests/stree_load_test.sh
    TEST=get_global_aabb_sibling_overlaps ~/o/sysrap/tests/stree_load_test.sh pdb


Morton code based approach
---------------------------

The "simtrace" intersects use a quad4 struct with 4x4 elements with intersect positions, normals and identity integers.

Please generate some CUDA Thrust based code to:

1. copy from full buffer of intersect quad4 to a subset within an input bbox
2. compute Morton codes for intersect positions within the subset bbox
3. sort the quad4 structs by their Morton codes
4. select intersects from the sorted ones with identity patterns suggesting overlap
    (without overlaps would see contiguous identity integers with clean transitions to other identities : how to select the converse of that with mixed identities ? )



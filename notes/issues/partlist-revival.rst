partlist-revival
==================

Suspect reviving partlist would work well for PMTs 
as after avoiding the separator anti-pattern they 
can be modelled by a "unionlist".

For code coherency that means reimplimenting 
the partlist branch of intersect_analytic to use
the same base bounds and intersects the full-blown CSG does::

    //csg_intersect_part.h 
    void csg_bounds_prim(int primIdx, const Prim& prim, optix::Aabb* aabb )
    void csg_intersect_part(const Prim& prim, const unsigned partIdx, const float& tt_min, float4& tt  )

Hmm, could solids that partlist would work with be auto-detected ?
If not could control it via an option based on solid names.


**Contrary to the old partlist it is better to make more use of the 
BVH by using bbox for each of the partitioned single primIdx.**

* see NTreeChopper : originally thinking about detecting Inner_Separator : but 
  instead did that at source with "--pmt20inch-simplify-csg"



  



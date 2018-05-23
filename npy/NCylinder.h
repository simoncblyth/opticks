#pragma once

// TODO: eliminate this, as CSG solids must be closed anyhow
// this avoids duplication between NCylinder.hpp  and oxrap/cu/csg_intersect_primitive.h
enum {
   CYLINDER_ENDCAP_P = 0x1 << 0,
   CYLINDER_ENDCAP_Q = 0x1 << 1
};


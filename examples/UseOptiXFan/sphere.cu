#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float4, sphere_param, , );
rtDeclareVariable(float3, hit_pos, attribute hit_pos, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

template <bool use_robust_method>
static __device__ void intersect_sphere(void)
{
  float3 center = make_float3(sphere_param);
  float3 O = ray.origin - center;
  float l = 1.f / length(ray.direction);
  float3 D = ray.direction * l;
  float radius = sphere_param.w;

  float b = dot(O, D);
  float c = dot(O, O) - radius * radius;
  float disc = b * b - c;
  if (disc > 0.0f)
  {
    float sdisc = sqrtf(disc);
    float root1 = (-b - sdisc);
    bool do_refine = false;
    float root11 = 0.0f;
    if (use_robust_method && fabsf(root1) > 10.f * radius)
      do_refine = true;
    if (do_refine)
    {
      // refine root1
      float3 O1 = O + root1 * D;
      b = dot(O1, D);
      c = dot(O1, O1) - radius * radius;
      disc = b * b - c;

      if (disc > 0.0f)
      {
        sdisc = sqrtf(disc);
        root11 = (-b - sdisc);
      }
    }

    bool check_second = true;
    if (rtPotentialIntersection((root1 + root11) * l))
    {
      hit_pos = center;
      if (rtReportIntersection(0)) //? why rtReportIntersection can fail?
        check_second = false;
    }
    if (check_second)
    {
      float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
      if (rtPotentialIntersection(root2 * l))
      {
        hit_pos = center;
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void intersect(int primIdx)
{
    intersect_sphere<true>();
}

RT_PROGRAM void bounds(int, float result[6])
{
    const float3 cen = make_float3(sphere_param);
    const float3 rad = make_float3(sphere_param.w);

    rtPrintf("// bounds cen (%f %f %f) radius %f \n", cen.x, cen.y, cen.z, rad.x );

    optix::Aabb *aabb = (optix::Aabb *)result;

    if (rad.x > 0.0f && !isinf(rad.x))
    {
        aabb->m_min = cen - rad;
        aabb->m_max = cen + rad;
    }
    else
        aabb->invalidate();
}

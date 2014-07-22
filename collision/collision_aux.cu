
/*
* Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#include <optix_world.h>
#include "helpers.h"

using namespace optix;

struct PerRayData_collision
{
  float range;
};

rtBuffer<float4, 1>           moving_objs0; // Even frames: Input positions; Odd frames: Output positions
rtBuffer<float4, 1>           moving_objs1; // Even frames: Output positions; Odd frames: Input positions
rtBuffer<float3, 1>           probe_dirs;
rtBuffer<float, 1>            output_probe_buffer; // X: probe directions; Y: moving objects
rtBuffer<float, 1>            output_LOS_buffer; // X: Target; Y: Viewer

rtDeclareVariable(uint,       moving_obj_count, ,); // How many objects there are this frame.
rtDeclareVariable(int,        swapped, ,);
rtDeclareVariable(int,        simMode, ,);
rtDeclareVariable(float,      collision_distance, ,);
rtDeclareVariable(float,      target_speed, ,);
rtDeclareVariable(float,      bad_range, ,);
rtDeclareVariable(float,      miss_range, ,);
rtDeclareVariable(float,      scene_epsilon, ,);
rtDeclareVariable(float,      time_view_scale, ,) = 1e-6f;
rtDeclareVariable(rtObject,   top_object, ,);

rtDeclareVariable(uint,       launch_index, rtLaunchIndex,);
rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
rtDeclareVariable(float,      t_hit, rtIntersectionDistance,);
rtDeclareVariable(PerRayData_collision, prd_collision, rtPayload,);

//#define TIME_VIEW

RT_PROGRAM void collision_any_hit()
{
  prd_collision.range = t_hit;
  rtTerminateRay();
}

RT_PROGRAM void collision_closest_hit()
{
  prd_collision.range = t_hit;
}

RT_PROGRAM void collision_raygen()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif

  // X: probe directions; Y: moving objects
  unsigned int probei = launch_index % probe_dirs.size();
  unsigned int vieweri = launch_index / probe_dirs.size();

  float3 ray_origin = make_float3(swapped ? moving_objs1[vieweri] : moving_objs0[vieweri]);
  float3 ray_direction = probe_dirs[probei];

  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, collision_distance);

  PerRayData_collision prd;
  prd.range = miss_range;

  rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
  clock_t t1 = clock(); 

  float expected_fps   = 1.0f;
  float pixel_time     = (t1 - t0) * time_view_scale * expected_fps;
  output_probe_buffer[launch_index] = pixel_time; 
#else
  output_probe_buffer[launch_index] = prd.range;
#endif
}

RT_PROGRAM void collision_exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf("Caught exception 0x%X at launch index %d\n", code, launch_index);
  output_probe_buffer[launch_index] = bad_range;
}

RT_PROGRAM void line_of_sight_raygen()
{
  // X: Target; Y: Viewer
  unsigned int targeti = launch_index % moving_obj_count;
  unsigned int vieweri = launch_index / moving_obj_count;

  if(vieweri < targeti) { // Don't trace the same ray in both directions.
#ifdef TIME_VIEW
    clock_t t0 = clock(); 
#endif

    PerRayData_collision prd;
    prd.range = miss_range;

    float3 ray_origin = make_float3(swapped ? moving_objs1[vieweri] : moving_objs0[vieweri]);
    float3 target = make_float3(swapped ? moving_objs1[targeti] : moving_objs0[targeti]);
    float3 ray_direction = target - ray_origin;

    optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, 0, scene_epsilon, 1);

    rtTrace(top_object, ray, prd);

#ifdef TIME_VIEW
    clock_t t1 = clock(); 

    float expected_fps   = 1.0f;
    float pixel_time     = (t1 - t0) * time_view_scale * expected_fps;
    output_LOS_buffer[launch_index] = pixel_time; 
#else
    output_LOS_buffer[launch_index] = prd.range;
#endif
  }
}

RT_PROGRAM void line_of_sight_exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf("Caught exception 0x%X at launch index %d\n", code, launch_index);
  output_LOS_buffer[launch_index] = bad_range;
}

// Find the probe direction closest to the query direction
static __device__ uint GetMatchingDir(const float3 &in)
{
  float maxDot = -1e35f;
  uint bestPr = 0u;
  for(uint pr=0; pr<probe_dirs.size(); pr++) {
    float d = dot(probe_dirs[pr], in);
    if(d>maxDot) {
      maxDot = d;
      bestPr = pr;
    }
  }

  return bestPr;
}

static __device__ float3 collision_reduction()
{
  uint v = launch_index;

  float3 V = make_float3(swapped ? moving_objs1[v] : moving_objs0[v]);

  // Find the closest collision direction and move away from it
  float bestR = miss_range;
  size_t bestPr = 0u;

  for(size_t pr=0; pr<probe_dirs.size(); pr++) {
    float Range = output_probe_buffer[v*probe_dirs.size()+pr];
    if(Range < bestR) {
      bestR = Range;
      bestPr = pr;
    }
  }

  float3 V0 = make_float3(swapped ? moving_objs0[v] : moving_objs1[v]);
  float3 NewVel = V - V0; // Current velocity, based on old position

  if(bestR < collision_distance) { // Imminent collision
    NewVel = -probe_dirs[bestPr];
    float mySpeed = min(bestR, target_speed); // We're going in the opposite direction of bestR, but limiting our speed makes sure we don't go through an opposite wall.
    NewVel = normalize(NewVel) * mySpeed;
  } 
  // XXX Do something interesting sometimes

  float3 VOut = V + NewVel;

  return VOut;
}

static __device__ float3 LOS_reduction()
{
  uint v = launch_index;

  float3 V = make_float3(swapped ? moving_objs1[v] : moving_objs0[v]);

  // Find a target to run from or toward
  float extremeTDstSqr = 1e35f;
  float3 TargetDir;
  uint width = moving_obj_count;
  for(size_t t=0; t<width; t++) {
    if(v==t) continue;

    uint coords = (v<t) ? (v*moving_obj_count+t) : (t*moving_obj_count+v);
    float cdst = output_LOS_buffer[coords]; // Keep in the upper diagonal
    if(cdst != miss_range) continue; // Can't see this target

    float3 T = make_float3(swapped ? moving_objs1[t] : moving_objs0[t]);
    float3 dir = T - V; // Direction to target
    float tdstSqr = dot(dir,dir); // Distance to target

    if(tdstSqr < extremeTDstSqr) {
      extremeTDstSqr = tdstSqr;
      if(v&1) { // Odd: Green:
        if(t&1)
          TargetDir = -dir; // Go away from closest guy in view if it's green
        else
          TargetDir = dir; // Go toward closest guy in view if it's red
      } else { // Even: Red:
        if(t&1)
          TargetDir = -dir; // Go away from closest guy in view if it's green
        else
          TargetDir = -dir; // Go away from closest guy in view if it's red
      }
    }
  }

  // Can't acquire a target so try to keep moving forward
  if(extremeTDstSqr == 1e35f) {
    float3 V0 = make_float3(swapped ? moving_objs0[v] : moving_objs1[v]); // Get my old position
    TargetDir = V - V0;
  }

  // We must be careful to only go in EXACTLY a direction we've probed,
  // not in a TargetDir. This includes not setting the z component to 0.

  // See if there is a wall in the target direction
  uint pro = GetMatchingDir(TargetDir);
  float Range = output_probe_buffer[v*probe_dirs.size()+pro];
  float3 MatchingDir = probe_dirs[pro];

  if(Range > collision_distance) {
    // Plenty of room ahead; go for it
    TargetDir = probe_dirs[pro];
  } else {
    // Find the closest direction to the target that has no collision

    // This can press us into but not through walls.
    // We can step diagonally toward the wall. But why don't we then go through it?
    // Because there are no unoccluded probes in the wall's half space.
    float maxDot = -1e35f;
    size_t bestPr = 0u;
    for(size_t pr=0; pr<probe_dirs.size(); pr++) {
      float d = dot(probe_dirs[pr], TargetDir);
      float Range = output_probe_buffer[v*probe_dirs.size()+pr];
      if(Range == miss_range && d > maxDot) {
        maxDot = d;
        bestPr = pr;
      }
    }

    TargetDir = probe_dirs[bestPr];
  }

  TargetDir = normalize(TargetDir);
  float3 NewVel = TargetDir * ((v&1) ? target_speed : target_speed*1.2f);
  float3 VOut = V + NewVel;

  return VOut;
}

RT_PROGRAM void reduction_raygen()
{
#ifdef TIME_VIEW
  clock_t t0 = clock(); 
#endif

  float3 VOut = simMode == 1 ? LOS_reduction() : collision_reduction();

#ifdef TIME_VIEW
  clock_t t1 = clock(); 

  float expected_fps = 1.0f;
  float pixel_time   = (t1 - t0) * time_view_scale * expected_fps;
  VOut = make_float3(pixel_time);
#endif

  if(swapped)
    moving_objs0[launch_index] = make_float4(VOut,1.0f);
  else
    moving_objs1[launch_index] = make_float4(VOut,1.0f);
}

RT_PROGRAM void reduction_exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf("Caught exception 0x%X at launch index (%d)\n", code, launch_index);
  if(swapped)
    moving_objs0[launch_index] = make_float4(bad_range);
  else
    moving_objs1[launch_index] = make_float4(bad_range);
}

// Don't use this because there's a bug where it calls the miss program after rtTerminateRay().
// Instead, init to the miss_range.
RT_PROGRAM void collision_miss()
{
  prd_collision.range = miss_range;
}

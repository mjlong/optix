
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

#include <stdlib.h>
#include <stdio.h>


#include "../../include/optix_world.h"

using namespace optix;

// Find the probe direction closest to the query direction
__device__ uint GetMatchingDir(const float3 *probe_dirs, uint probe_dirs_size, const float3 &in)
{
  float maxDot = -1e35f;
  uint bestPr = 0u;
  for(uint pr=0; pr<probe_dirs_size; pr++) {
    float d = dot(probe_dirs[pr], in);
    if(d>maxDot) {
      maxDot = d;
      bestPr = pr;
    }
  }

  return bestPr;
}

extern "C" __global__ void LOS_reduction( const float4 *moving_objs, float4 *moving_objs_other_frame,
                                          uint moving_obj_count,
                                          const float3 *probe_dirs, uint probe_dirs_size,
                                          const float *output_LOS_buffer, const float *output_probe_buffer,
                                          float collision_distance, float miss_range, float target_speed )
{
  uint v = blockIdx.x;

  float3 V = optix::make_float3( moving_objs[v] );

  // Find a target to run from or toward
  float extremeTDstSqr = 1e35f;
  float3 TargetDir;
  uint width = moving_obj_count;
  for(size_t t=0; t<width; t++) {
    if(v==t) continue;

    uint coords = (v<t) ? (v*moving_obj_count+t) : (t*moving_obj_count+v);
    float cdst = output_LOS_buffer[coords]; // Keep in the upper diagonal
    if(cdst != miss_range) continue; // Can't see this target

    float3 T = optix::make_float3( moving_objs[t] );
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
    float3 V0 = optix::make_float3( moving_objs_other_frame[v] ); // Get my old position
    TargetDir = V - V0;
  }

  // We must be careful to only go in EXACTLY a direction we've probed,
  // not in a TargetDir. This includes not setting the z component to 0.

  // See if there is a wall in the target direction
  uint pro = GetMatchingDir(probe_dirs, probe_dirs_size, TargetDir);
  float Range = output_probe_buffer[v*probe_dirs_size+pro];

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
    for(size_t pr=0; pr<probe_dirs_size; pr++) {
      float d = dot(probe_dirs[pr], TargetDir);
      float Range = output_probe_buffer[v*probe_dirs_size+pr];
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

    moving_objs_other_frame[v] = optix::make_float4(VOut,1.0f);
}


extern "C" __host__ void LOS_reduction_CUDA( const float4 *moving_objs, float4 *moving_objs_other_frame,
                                             uint moving_obj_count,
                                             const float3 *probe_dirs, uint probe_dirs_size,
                                             const float *output_LOS_buffer, const float *output_probe_buffer,
                                             float collision_distance, float miss_range, float target_speed )
{
  LOS_reduction <<< moving_obj_count, 1 >>> ( moving_objs, moving_objs_other_frame,
                                              moving_obj_count,
                                              probe_dirs, probe_dirs_size,
                                              output_LOS_buffer,
                                              output_probe_buffer,
                                              collision_distance, miss_range, target_speed );
}




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

// CollisionOptiX.h
//
// The OptiX-related and simulation-related pieces of the collision detection sample

#ifndef _collision_optix_h
#define _collision_optix_h

#include "PointsOnSphere.h"
#include "MovingObjects.h"

#include <nvModel.h>
#include <optix_world.h>

enum SimModes_E {
  SIM_OBJECTS_AVOID_WALLS = 0,
  SIM_LINE_OF_SIGHT_EYE = 1,

  SIM_NUM_MODES = 2
};

enum {
  CollisionInd = 0,
  LOSInd = 1,
  ReductionInd = 2
};

class CollisionScene
{
public:
  CollisionScene(size_t num_moving_objects, size_t num_probes, const std::string& filename, const optix::Aabb &initial_aabb, SimModes_E m_simMode,
    const unsigned int modelVB, const unsigned int modelIB, nv::Model *OGLModel) : simMode(m_simMode), frame(0)
      {
        initScene(num_moving_objects, num_probes, filename, initial_aabb, modelVB, modelIB, OGLModel);
      }

      void initScene(size_t num_moving_objects, size_t num_probes, const std::string& filename, const optix::Aabb &initial_aabb,
        const unsigned int modelVB, const unsigned int modelIB, nv::Model *OGLModel); // stuff the geometry into OptiX
      void updateScene(bool updMovingObjs = true, size_t moving_obj_count=0); // call each frame to animate stuff and do collision detection
      void cleanUp(); // delete it all
      float *getOutputBuffer(const char *buf_name);
      void unmapOutputBuffer(const char *buf_name);
      void updateMovingObjs(size_t moving_obj_count); // Read the new object locations from the buffer into MovingObjs.

      SimModes_E getSimMode() { return simMode; }

private:
  void SimObjectsAvoidWalls(bool updMovingObjs = true, size_t moving_obj_count=0);
  void SimLineOfSightEye(bool updMovingObjs = true, size_t moving_obj_count=0);

  optix::Context Ctx;
  SimModes_E simMode;

public:
  PointsOnSphere ProbeDirs;
  MovingObjects MovingObjs;

  float targetSpeed;
  float collisionDistance;
  float objectDiameter;
  int frame;
  bool horizontal; // True to only allow movement in X and Y. Useful for mazes.

  static const float BAD_RANGE;
  static const float MISS_RANGE;
};

#endif

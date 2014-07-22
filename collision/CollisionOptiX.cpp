
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

#include "CollisionOptiX.h"

#include <ObjLoader.h>

#include <optix_world.h>
#include <cuda_runtime.h>

#include <limits>
#include <cstring>

namespace {
  const char* const ptxpath(const std::string& base)
  {
    static std::string path;
    path = std::string(sutilSamplesPtxDir()) + "/collision_generated_" + base + ".ptx";
    return path.c_str();
  }
};

const float CollisionScene::BAD_RANGE = std::numeric_limits<float>::quiet_NaN();
const float CollisionScene::MISS_RANGE = std::numeric_limits<float>::max();

void CollisionScene::initScene(size_t num_moving_objects, size_t num_probes, const std::string& filename, const optix::Aabb &initial_aabb,
                               const unsigned int modelVB, const unsigned int modelIB, nv::Model *OGLModel)
{    
  targetSpeed = 0.07f;
  collisionDistance = 0.4f;
  objectDiameter = 0.3f;
  horizontal = initial_aabb.extent(2) < 3.0f; // Detect mazes

  // context
  Ctx = Context::create();
  Ctx->setStackSize(320);
  Ctx["bad_range"]->setFloat(BAD_RANGE);
  Ctx["miss_range"]->setFloat(MISS_RANGE);
  Ctx["scene_epsilon"]->setFloat(1.e-4f);
  Ctx["collision_distance"]->setFloat(collisionDistance);
  Ctx["target_speed"]->setFloat(targetSpeed);
  Ctx["simMode"]->setInt(simMode);
  Ctx["swapped"]->setInt(0);
  Ctx["moving_obj_count"]->setUint(0);

  Ctx->setRayTypeCount(2);
  Ctx->setEntryPointCount(3);
  Ctx->setPrintEnabled(true);
  Ctx->setPrintBufferSize(1024);

  // Collision ray generation, miss, and exception program
  std::string ptx_path = ptxpath("collision_aux.cu");
  Ctx->setRayGenerationProgram(CollisionInd, Ctx->createProgramFromPTXFile(ptx_path, "collision_raygen"));
  Ctx->setRayGenerationProgram(LOSInd, Ctx->createProgramFromPTXFile(ptx_path, "line_of_sight_raygen"));
  Ctx->setRayGenerationProgram(ReductionInd, Ctx->createProgramFromPTXFile(ptx_path, "reduction_raygen"));
  Ctx->setExceptionProgram(CollisionInd, Ctx->createProgramFromPTXFile(ptx_path, "collision_exception"));
  Ctx->setExceptionProgram(LOSInd, Ctx->createProgramFromPTXFile(ptx_path, "line_of_sight_exception"));
  Ctx->setExceptionProgram(ReductionInd, Ctx->createProgramFromPTXFile(ptx_path, "reduction_exception"));

  // Fill the probe direction buffer
  ProbeDirs.MakeRandomPoints(num_probes, horizontal);
  Buffer probe_dir_buffer = Ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_probes);
  float3* probe_dirs = reinterpret_cast<float3*>(probe_dir_buffer->map());
  memcpy(probe_dirs, &(ProbeDirs[0]), sizeof(float3) * ProbeDirs.size());
  probe_dir_buffer->unmap();
  Ctx["probe_dirs"]->set(probe_dir_buffer);

  // The box in which to generate the moving object positions
  optix::Aabb AB;
  AB.m_min = (initial_aabb.m_min - initial_aabb.center()) * 0.5f + initial_aabb.center();
  AB.m_max = (initial_aabb.m_max - initial_aabb.center()) * 0.5f + initial_aabb.center();
  if(horizontal) AB.m_max.z = AB.m_min.z;
  std::cerr << AB.m_min << AB.m_max << '\n';

  // The buffers for the centers of the moving objects. These will swap each frame.
  MovingObjs.MakeRandomObjects(num_moving_objects, AB);
  Buffer moving_objs0 = Ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, num_moving_objects);
  Buffer moving_objs1 = Ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, num_moving_objects);
  float4* moving_objs0P = reinterpret_cast<float4*>(moving_objs0->map());
  float4* moving_objs1P = reinterpret_cast<float4*>(moving_objs1->map());
  for(size_t v=0; v<num_moving_objects; v++) {
    moving_objs0P[v] = make_float4(MovingObjs[v], 1.0f);
    moving_objs1P[v] = make_float4(MovingObjs[v]+MovingObjs.Vels[v], 1.0f);
  }
  moving_objs0->unmap();
  moving_objs1->unmap();
  Ctx["moving_objs0"]->set(moving_objs0);
  Ctx["moving_objs1"]->set(moving_objs1);

  // The buffers for the ray intersection results
  // X: probe directions; Y: moving objects
  Buffer output_probe_buffer = Ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, num_moving_objects * num_probes);
  Ctx["output_probe_buffer"]->set(output_probe_buffer);
  // SIM_LINE_OF_SIGHT_EYE: X: Target; Y: Viewer
  Buffer output_LOS_buffer = Ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, num_moving_objects * num_moving_objects);
  Ctx["output_LOS_buffer"]->set(output_LOS_buffer);

  Material CollisionMat = Ctx->createMaterial();
  // To avoid walls we need to know how far away everything is.
  CollisionMat->setClosestHitProgram(CollisionInd, Ctx->createProgramFromPTXFile(ptx_path, "collision_closest_hit"));
  // For line of sight we just need to know whether it's occluded.
  CollisionMat->setAnyHitProgram(LOSInd, Ctx->createProgramFromPTXFile(ptx_path, "collision_any_hit"));

  Acceleration AS = Ctx->createAcceleration("Sbvh","Bvh");

  // Bind the static environment model into OptiX from its OpenGL buffers.
  Geometry G = Ctx->createGeometry();

  G->setIntersectionProgram(Ctx->createProgramFromPTXFile(ptxpath("triangle_mesh_fat.cu"), "mesh_intersect"));
  G->setBoundingBoxProgram(Ctx->createProgramFromPTXFile(ptxpath("triangle_mesh_fat.cu"), "mesh_bounds"));

  int num_prims = OGLModel->getCompiledIndexCount()/3;
  int vert_size = OGLModel->getCompiledVertexSize();
  int num_vertices = OGLModel->getCompiledVertexCount();
  G->setPrimitiveCount(num_prims);
  std::cerr << "num_prims=" << num_prims << " num_vertices=" << num_vertices << " vert_size=" << vert_size << '\n';

  Buffer vertex_buffer = Ctx->createBufferFromGLBO(RT_BUFFER_INPUT, modelVB);
  vertex_buffer->setFormat(RT_FORMAT_USER);
  vertex_buffer->setElementSize(vert_size*sizeof(float)); // X,Y,Z,Nx,Ny,Nz
  vertex_buffer->setSize(num_vertices);
  G["vertex_buffer"]->setBuffer(vertex_buffer);

  Buffer index_buffer = Ctx->createBufferFromGLBO(RT_BUFFER_INPUT, modelIB);
  index_buffer->setFormat(RT_FORMAT_INT3);
  index_buffer->setSize(num_prims);
  G["index_buffer"]->setBuffer(index_buffer);

  Buffer material_buffer = Ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, num_prims);
  void* material_data = material_buffer->map();
  memset(material_data, 0, num_prims*sizeof(unsigned int));
  material_buffer->unmap();
  G["material_buffer"]->setBuffer(material_buffer);

  AS->setProperty("vertex_buffer_stride", "24");

  GeometryInstance GI = Ctx->createGeometryInstance(G, &CollisionMat, (&CollisionMat)+1);
  GeometryGroup StaticGG = Ctx->createGeometryGroup(&GI, (&GI)+1);

  StaticGG->setAcceleration(AS);

  Ctx["top_object"]->set(StaticGG);

  // Prepare to run
  Ctx->validate();
  Ctx->compile();
}

// get the buffer that has the collision results
float *CollisionScene::getOutputBuffer(const char *buf_name)
{
  Buffer B = Ctx[buf_name]->getBuffer();

  float *ranges = static_cast<float *>(B->map());
  return ranges;
}

void CollisionScene::unmapOutputBuffer(const char *buf_name)
{
  Ctx[buf_name]->getBuffer()->unmap();
}

// Look at returned collision ranges and update MovingObjs accordingly
void CollisionScene::SimObjectsAvoidWalls(bool updMovingObjs, size_t moving_obj_count)
{
  Buffer PB = Ctx["output_probe_buffer"]->getBuffer();
  RTsize PB_size;
  PB->getSize(PB_size);
  assert(MovingObjs.size()*ProbeDirs.size() == PB_size);
  Ctx->launch(CollisionInd, static_cast<unsigned int>(moving_obj_count) * static_cast<unsigned int>(ProbeDirs.size()));

  // Do the dynamics on the GPU
  Ctx->launch(ReductionInd, static_cast<unsigned int>(moving_obj_count));
}

// import the CUDA routine for object movement update
// NB: to link correctly, this will need the accompanying collision_CUDA.cu to be compiled as native CUDA code
extern "C" void LOS_reduction_CUDA( const float4 *moving_objs, float4 *moving_objs_other_frame,
                                    uint moving_obj_count,
                                    const float3 *probe_dirs, uint probe_dirs_size,
                                    const float *output_LOS_buffer, const float *output_probe_buffer,
                                    float collision_distance, float miss_range, float target_speed );

// helper function to extract device pointer
template <class T>
T rtGetBDP( Context& context, RTbuffer buf, int optixDeviceIndex )
{
  void* bdp;
  RTresult res = rtBufferGetDevicePointer(buf, optixDeviceIndex, &bdp);
  if ( RT_SUCCESS != res )
  {
    sutilHandleErrorNoExit( context->get(), res, __FILE__, __LINE__ );
  }
  return (T) bdp;
}

// helper function to obtain the CUDA device ordinal of a given OptiX device
int GetOptixDeviceOrdinal( const Context& context, unsigned int optixDeviceIndex )
{
  unsigned int numOptixDevices = context->getEnabledDeviceCount();
  std::vector<int> devices = context->getEnabledDevices();
  int ordinal;
  if ( optixDeviceIndex < numOptixDevices )
  {
    context->getDeviceAttribute( devices[optixDeviceIndex], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal );
    return ordinal;
  }
  return -1;
}


// Look at returned line-of-sight ranges and update MovingObjs accordingly
void CollisionScene::SimLineOfSightEye(bool updMovingObjs, size_t moving_obj_count)
{
  Buffer PB = Ctx["output_probe_buffer"]->getBuffer();
  RTsize PB_size;
  PB->getSize(PB_size);
  assert(MovingObjs.size()*ProbeDirs.size() == PB_size);
  Ctx->launch(CollisionInd, static_cast<unsigned int>(moving_obj_count) * static_cast<unsigned int>(ProbeDirs.size()));

  Buffer LB = Ctx["output_LOS_buffer"]->getBuffer();
  RTsize LB_size;
  LB->getSize(LB_size);
  assert(MovingObjs.size()*MovingObjs.size() == LB_size);
  Ctx->launch(LOSInd, static_cast<unsigned int>(moving_obj_count) * static_cast<unsigned int>(moving_obj_count));

  // Do the dynamics update on the GPU
  int firstOrdinal = GetOptixDeviceOrdinal( Ctx, 0 );

  if ( firstOrdinal >= 0 )
  {
    // Update the objects through a CUDA routine
    float4 *moving_objs0_buffer_device_ptr = rtGetBDP<float4*>(Ctx, Ctx["moving_objs0"]->getBuffer()->get(), 0);
    float4 *moving_objs1_buffer_device_ptr = rtGetBDP<float4*>(Ctx, Ctx["moving_objs1"]->getBuffer()->get(), 0);
  
    const float4 *moving_objs = const_cast<const float4*>(Ctx["swapped"]->getInt() ? moving_objs1_buffer_device_ptr
                                                                                   : moving_objs0_buffer_device_ptr );
    float4 *moving_objs_other_frame = Ctx["swapped"]->getInt() ? moving_objs0_buffer_device_ptr
                                                               : moving_objs1_buffer_device_ptr;
    const float3* probe_dirs_BDP = rtGetBDP<const float3*>(Ctx, Ctx["probe_dirs"]->getBuffer()->get(), 0);
    
    const float* output_LOS_BDP = rtGetBDP<const float*>(Ctx, Ctx["output_LOS_buffer"]->getBuffer()->get(), 0);
    const float* output_probe_BDP = rtGetBDP<const float*>(Ctx, Ctx["output_probe_buffer"]->getBuffer()->get(), 0);
    cudaSetDevice(firstOrdinal);
    LOS_reduction_CUDA( moving_objs, moving_objs_other_frame,
                        Ctx["moving_obj_count"]->getUint(),
                        probe_dirs_BDP, (uint)ProbeDirs.size(),
                        output_LOS_BDP, output_probe_BDP,
                        Ctx["collision_distance"]->getFloat(), Ctx["miss_range"]->getFloat(), Ctx["target_speed"]->getFloat() );

    // see if the kernel launch finished successfully
    cudaError_t lastError = cudaGetLastError();
    if ( cudaSuccess != lastError )
    {
      std::cerr << "LOS_reduction_CUDA reports an error: "<<cudaGetErrorString(lastError)<<"\n";
    }
  }
}

// Read back the object positions into MovingObjs
void CollisionScene::updateMovingObjs(size_t moving_obj_count)
{
  Buffer RB = (frame & 1) ? Ctx["moving_objs0"]->getBuffer() : Ctx["moving_objs1"]->getBuffer();
  RTsize RB_width;
  RB->getSize(RB_width);
  assert(MovingObjs.size() == RB_width);

  float4 *new_moving_objs = static_cast<float4 *>(RB->map());
  for(size_t v=0; v<moving_obj_count; v++)
    MovingObjs[v] = make_float3(new_moving_objs[v]);
  RB->unmap();
}

// moving_obj_count==0 means all of them.
void CollisionScene::updateScene(bool updMovingObjs, size_t moving_obj_count)
{
  if(moving_obj_count==0) moving_obj_count = MovingObjs.size();
  assert(moving_obj_count <= MovingObjs.size());

  Ctx["swapped"]->setInt(frame&1);
  frame++;
  MovingObjs.CurSize = moving_obj_count;
  Ctx["moving_obj_count"]->setUint(static_cast<unsigned int>(moving_obj_count));

  // Move the models based on collision results
  switch(simMode) {
    case SIM_OBJECTS_AVOID_WALLS:
      SimObjectsAvoidWalls(updMovingObjs, moving_obj_count);
      break;
    case SIM_LINE_OF_SIGHT_EYE:
      SimLineOfSightEye(updMovingObjs, moving_obj_count);
      break;
    default:
      assert(0);
      break;
  }

  if(updMovingObjs)
    updateMovingObjs(moving_obj_count);

}

void CollisionScene::cleanUp()
{
  Ctx->destroy();
}

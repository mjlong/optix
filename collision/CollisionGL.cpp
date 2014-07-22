
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

// CollisionGL.cpp
//
// A testbed for non-graphics simulation using ray tracing
// Collision detection, line-of-sight, path planning, etc.

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  else
#    include <GL/glxew.h>
#  endif
#  include <GL/glut.h>
#endif

#include "CollisionUtils.h"
#include "CollisionOptiX.h"

#include <ObjLoader.h>
#include <PPMLoader.h>
#include <sutil.h>
#include <nvModel.h>

#include <optix_world.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

namespace {
  int LastMouseX = 0, LastMouseY = 0;
  int Width = 1024, Height = 768;
  int OldWidth, OldHeight;
  int F_LitTex_Id = -1;
  bool FullScreen = false;
  bool VisualizeMatrices = false;
  bool benchmark_no_display = false;
  float RangeVisScale = 1.0f;
  float VFovDeg = 30.0f;
  float3 Eye = {12.7359f, -12.9758f, 49.4106f};
  float3 LookAt = {15.8922f, 15.0068f, -0.96137f};
  float3 Up = {-0.129181f, 0.87026f, 0.475352f};
  optix::Aabb WorldAABB;
  std::string AOHackfilename;
  CollisionScene *Scene = NULL;
  nv::Model *OGLModel;
  GLuint modelVB;
  GLuint modelIB;
  SimModes_E simMode = SIM_LINE_OF_SIGHT_EYE;
  unsigned int warmup_frames = 10u, timed_frames = 10u;
};

void idle()
{
  // Trigger a call to display(), which will update everything
  glutPostRedisplay();
}

void reshape(int w, int h)
{
  Width = w;
  Height = h;
}

const char V_LitTex[]=
{
  "!!VP1.0\n"
  // Transform vertex into clip-space using modelview-projection matrix
  "DP4 o[HPOS].x, v[OPOS], c[4];"
  "DP4 o[HPOS].y, v[OPOS], c[5];"
  "DP4 o[HPOS].z, v[OPOS], c[6];"
  "DP4 o[HPOS].w, v[OPOS], c[7];"

  // Local light
  "ADD o[TEX1], -v[OPOS], c[2];" // TEX2 = LightPos - VertexPos
  "MOV o[TEX0], v[NRML];" // Vertex normal
  "MOV o[COL0], v[COL0];" // Vertex color
  "\nEND"
};

const char F_LitTex[]=
{
  // TEX0 - vertex normal (not normalized except at vertices)
  // TEX1 - vec to light (not normalized)

  // Variables:
  "!!FP1.0\n"
  "DECLARE LightCol;\n"

  "DP3 R3, f[TEX0], f[TEX0];"
  "RSQ R3, R3.x;"
  "MUL R3, f[TEX0], R3.xxxx;" // R3 Normalize interpolated normal

  "DP3 R2, f[TEX1], f[TEX1];"
  "RSQ R2, R2.x;"
  "MUL R2, f[TEX1], R2.xxxx;" // R2 Normalize light vector

  "DP3 R4, R2, R3;"     // N dot L
  "MAX R4, R4, -R4;"    // Negate the normal if necessary
  "ADD R4, R4, {0.3,0.3,0.3,0.3};" // Add some ambient irradiance
  "MUL R5, LightCol, R4.xxxx;"
  "MUL o[COLR], R5, f[COL0];"
  "\nEND"
};

static unsigned int LoadShaderProgram(GLenum ProgType, const char *prog)
{
  assert(prog && strlen((const char *)prog) > 10);
  assert(prog[0] == '!' && prog[1] == '!');
  GL_ASSERT();

  // XXX Don't leak these!!!
  unsigned int progID;
  glGenProgramsNV(1, &progID);
  assert(progID > 0);

  glBindProgramNV(ProgType, progID);
  GL_ASSERT();

  glLoadProgramNV(ProgType, progID, GLsizei(strlen((char *) prog)), (const unsigned char *)prog);
  GL_ASSERT();
  assert(glIsProgramNV(progID));

  return progID;
}

bool InitProgs()
{
  LoadShaderProgram(GL_VERTEX_PROGRAM_NV, V_LitTex);
  F_LitTex_Id = LoadShaderProgram(GL_FRAGMENT_PROGRAM_NV, F_LitTex);

  glEnable(GL_VERTEX_PROGRAM_NV);
  glEnable(GL_FRAGMENT_PROGRAM_NV);
  glEnable(GL_MULTISAMPLE);

  GL_ASSERT();

  int texnum = 0;
  glActiveTextureARB(GL_TEXTURE0_ARB + texnum);

  glBindTexture(GL_TEXTURE_2D, 2);
  GL_ASSERT();

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

  glPixelStorei(GL_UNPACK_ALIGNMENT,1);
  glPixelStorei(GL_PACK_ALIGNMENT,1);

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  return true;
}

GLuint MakeObjectDList(float Scale)
{
  GLuint ObjDList = glGenLists(1);
  glNewList(ObjDList, GL_COMPILE);
  //glRotatef(90, 1, 0, 0); glutSolidTeapot(0.8*Scale);
  //glutSolidTorus(0.7f*Scale*0.5f,0.7f*Scale,10,10);
  //glScalef(1.3f*Scale, 1.3f*Scale, 1.3f*Scale); glutSolidOctahedron();
  glScalef(0.618f*Scale, 0.618f*Scale, 0.618f*Scale); glutSolidDodecahedron();
  glEndList();

  return ObjDList;
}

void RenderMovingObjects()
{
  assert(Scene);

  float Scale = Scene->objectDiameter;

  static GLuint ObjDList = MakeObjectDList(Scale);

  for(size_t i=0; i<Scene->MovingObjs.CurSize; i++) {

    if(Scene->getSimMode() == SIM_LINE_OF_SIGHT_EYE && (i&1))
      glColor3f(0.2f, 0.6f, 0.3f); // Odd: Green
    else
      glColor3f(0.9f, 0.3f, 0.3f); // Even: red

    glPushMatrix();

    glTranslatef(Scene->MovingObjs[i].x, Scene->MovingObjs[i].y, Scene->MovingObjs[i].z);

    glCallList(ObjDList);

    glPopMatrix();
  }
}

void RenderLinesOfSight()
{
  if(Scene->getSimMode() != SIM_LINE_OF_SIGHT_EYE)
    return;

  float maxLen = WorldAABB.maxExtent();

  glDisable(GL_VERTEX_PROGRAM_NV);
  glDisable(GL_FRAGMENT_PROGRAM_NV);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_SRC_ALPHA);
  glLineWidth(2);

  glBegin(GL_LINES);

  size_t num_moving_objects = Scene->MovingObjs.CurSize;
  float *LOS_ranges = Scene->getOutputBuffer("output_LOS_buffer");
  for(size_t v=0; v<num_moving_objects; v++) {
    for(size_t t=0; t<num_moving_objects; t++) {
      if(v < t) {
        // X: Target; Y: Viewer
        float cdst = LOS_ranges[v*num_moving_objects+t];
        if(cdst == Scene->MISS_RANGE) {
          float3 &V = Scene->MovingObjs[v];
          float3 &T = Scene->MovingObjs[t];
          // cerr << "Line: " << V << T << endl;
          float len = length(T - V);
          float4 col = lerp(make_float4(0.3f,0.3f,0.6f,0.2f), make_float4(0.3f,0.3f,1.0f,0.2f), 1.0f - len/maxLen);
          glColor4f(col.x, col.y, col.z, col.w);
          glVertex3fv(reinterpret_cast<const GLfloat *>(&V));
          glVertex3fv(reinterpret_cast<const GLfloat *>(&T));
        }
      }
    }
  }

  // Draw probe rays
  size_t num_probes = Scene->ProbeDirs.size();
  float *probe_ranges = Scene->getOutputBuffer("output_probe_buffer");
  for(size_t v=0; v<num_moving_objects; v++) {
    for(size_t p=0; p<num_probes; p++) {
      // X: probe directions; Y: moving objects
      float cdst = probe_ranges[v*num_probes+p];
      if(cdst == Scene->MISS_RANGE) {
        glColor4f(0,1,0,0.5);
      } else {
        glColor4f(1,0,0,0.5);
      }
      float3 &V = Scene->MovingObjs[v];
      float3 T = V + Scene->ProbeDirs[p] * Scene->collisionDistance;
      glVertex3fv(reinterpret_cast<const GLfloat *>(&V));
      glVertex3fv(reinterpret_cast<const GLfloat *>(&T));
    }
  }

  glEnd();

  glDisable(GL_BLEND);

  Scene->unmapOutputBuffer("output_probe_buffer");
  Scene->unmapOutputBuffer("output_LOS_buffer");
}

// Render a visualization of the collisions
void RenderAOBackground()
{
  if(AOHackfilename.empty()) return;

  static PPMLoader PPM(AOHackfilename, true);

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_VERTEX_PROGRAM_NV);
  glDisable(GL_FRAGMENT_PROGRAM_NV);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glOrtho(0, Width, 0, Height, -1, 1);

  glRasterPos3f(0, 0, 0);
  glDrawPixels(PPM.width(), PPM.height(), GL_RGB, GL_UNSIGNED_BYTE, PPM.raster());
}

// Render a visualization of the collisions
void RenderCollisionBuffer()
{
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_VERTEX_PROGRAM_NV);
  glDisable(GL_FRAGMENT_PROGRAM_NV);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glOrtho(0, Width, 0, Height, -1, 1);

  size_t ofs = 0;
  {
    // X: probe directions; Y: moving objects
    size_t wid = Scene->ProbeDirs.size();
    size_t hgt = Scene->MovingObjs.CurSize;
    float *ranges = Scene->getOutputBuffer("output_probe_buffer");
    float *scaled_ranges = new float[wid*hgt];
    for(size_t i=0; i<wid*hgt; i++)
      scaled_ranges[i] = ranges[i] * RangeVisScale;
    glRasterPos3f(0, 0, 0);
    glDrawPixels( static_cast<GLsizei>( wid ), static_cast<GLsizei>( hgt ), GL_LUMINANCE, GL_FLOAT, scaled_ranges);
    delete [] scaled_ranges;
    Scene->unmapOutputBuffer("output_probe_buffer");
    ofs = wid + 10;
  }

  if(Scene->getSimMode() == SIM_LINE_OF_SIGHT_EYE) {
    // X: Viewer Y: Target
    size_t wid = Scene->MovingObjs.CurSize;
    size_t hgt = Scene->MovingObjs.CurSize;
    float *ranges = Scene->getOutputBuffer("output_LOS_buffer");
    float *scaled_ranges = new float[wid*hgt];
    for(size_t i=0; i<wid*hgt; i++)
      scaled_ranges[i] = ranges[i] * RangeVisScale;
    glRasterPos3f(float(ofs), 0, 0);
    glDrawPixels( static_cast<GLsizei>( wid ), static_cast<GLsizei>( hgt ), GL_LUMINANCE, GL_FLOAT, scaled_ranges);
    delete [] scaled_ranges;
    Scene->unmapOutputBuffer("output_LOS_buffer");
  }
}

void RenderStaticScene()
{
  glBindBuffer(GL_ARRAY_BUFFER, modelVB);
  glEnableClientState(GL_VERTEX_ARRAY);

  glVertexPointer(OGLModel->getPositionSize(), GL_FLOAT,
    OGLModel->getCompiledVertexSize()*sizeof(float),
    (void*) (OGLModel->getCompiledPositionOffset()*sizeof(float)));

  if (OGLModel->hasNormals()) {
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT,
      OGLModel->getCompiledVertexSize()*sizeof(float),
      (void*) (OGLModel->getCompiledNormalOffset()*sizeof(float)));
  }

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, modelIB);

  glDrawElements(GL_TRIANGLES, OGLModel->getCompiledIndexCount(), GL_UNSIGNED_INT, 0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_NORMAL_ARRAY);

  GL_ASSERT();
}

// Verify consistency between benchmark and non-benchmark mode by printing out some of the results of each.
// Normally this will be a no-op.
void VerifyResults()
{
  return;

  // X: Viewer Y: Target
  size_t wid = Scene->MovingObjs.CurSize;
  size_t hgt = Scene->MovingObjs.CurSize;
  float *ranges = Scene->getOutputBuffer("output_LOS_buffer");

  for(size_t i=0; i<wid*hgt; i++)
    if(ranges[i] != CollisionScene::MISS_RANGE) {
      std::cerr << i << ' ' << ranges[i] << ' ';
      break;
    }
  std::cerr << std::endl;

  Scene->unmapOutputBuffer("output_LOS_buffer");
}

void display()
{
  sutilFrameBenchmark("Collision", warmup_frames, timed_frames);

  assert(Scene);

  // Update the collision dynamics and motion
  Scene->updateScene();

  static bool Init = false;
  if(!Init) Init = InitProgs();

  glClearColor(0.3f, 0.3f, 0.3f,0);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  RenderAOBackground();

  glViewport(0, 0, Width, Height);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(VFovDeg, Width / double(Height), 1, 400);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt(Eye.x, Eye.y, Eye.z,
    LookAt.x, LookAt.y, LookAt.z,
    Up.x, Up.y, Up.z);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_VERTEX_PROGRAM_NV);
  glEnable(GL_FRAGMENT_PROGRAM_NV);

  glColor3f(0.9f, 0.9f, 0.9f);

  if(!AOHackfilename.empty()) {
    glColorMask(false, false, false, false);
  }

  RenderStaticScene();

  if(!AOHackfilename.empty()) {
    glColorMask(true, true, true, true);
  }

  //Load the Projection*Modelview matrix into registers c[4] to c[7].
  glTrackMatrixNV(GL_VERTEX_PROGRAM_NV, 4, GL_MODELVIEW_PROJECTION_NV, GL_IDENTITY_NV);

  float3 LightPos = Eye;
  LightPos.x += WorldAABB.maxExtent()*0.1f; // Place light in center of static scene
  glProgramParameter4fNV(GL_VERTEX_PROGRAM_NV, 1, Eye.x, Eye.y, Eye.z, 1); // Pass eye vector to vertex program
  glProgramParameter4fNV(GL_VERTEX_PROGRAM_NV, 2, LightPos.x, LightPos.y, LightPos.z, 1); // Pass light pos to vertex program
  GL_ASSERT();

  glProgramNamedParameter4fNV(F_LitTex_Id, 8, (unsigned char *)"LightCol", 1.0f, 1.0f, 1.0f, 1.0f);

  GL_ASSERT();

  RenderMovingObjects();

  RenderLinesOfSight();

  if(VisualizeMatrices)
    RenderCollisionBuffer();

  VerifyResults();

  glutSwapBuffers();
}

void Benchmark(unsigned int warmup_frames = 10u, unsigned int timed_frames = 10u)
{
  std::cerr << "Beginning benchmark.\n";

  assert(Scene);

  // Update the collision dynamics and motion
  for(unsigned int f=0; f<warmup_frames; f++) {
    Scene->updateScene(true); // Read positions back to host while we're not timing it.
    VerifyResults();
    std::cerr << '.';
  }

  std::cerr << "Starting clock.\n";
  std::cerr.flush();

  size_t moving_obj_count = 1, probe_dir_count = Scene->ProbeDirs.size();
  do {
    // Start timer and do the actual frames
    double start_frame_time = 0, end_frame_time = 0;
    sutilCurrentTime(&start_frame_time);

    for(unsigned int f=0; f<timed_frames; f++) {
      Scene->updateScene(false, moving_obj_count); // Don't read positions back to host while we're timing it.
      VerifyResults();
    }

    sutilCurrentTime(&end_frame_time);

    double seconds = end_frame_time - start_frame_time;
    double fps = double(timed_frames) / seconds;

    int rays = static_cast<int>( (moving_obj_count*moving_obj_count)/2 + moving_obj_count * probe_dir_count );
    double rps = double(rays) * double(timed_frames) / seconds;

    std::cerr << moving_obj_count << " objects. " << probe_dir_count << " probes. Ran " << timed_frames << " frames in " << seconds << " seconds. fps=" << fps << " " << rays << " rays, " << rps << " rays per second\n";
    moving_obj_count *= 2u;
  } while(moving_obj_count <= Scene->MovingObjs.size());
}

void Menu(int c, int x, int y)
{
  switch(c)
  {
  case 27:
  case 'q':
    exit(0);
    break;
  case '<':
    RangeVisScale *= 0.9f;
    std::cerr << RangeVisScale << '\n';
    break;
  case '>':
    RangeVisScale /= 0.9f;
    std::cerr << RangeVisScale << '\n';
    break;
  case 'v':
    VisualizeMatrices = !VisualizeMatrices;
    break;
  case 'f':
    FullScreen = !FullScreen;
    if(FullScreen)
    {
      OldWidth = glutGet(GLenum(GLUT_WINDOW_WIDTH));
      OldHeight = glutGet(GLenum(GLUT_WINDOW_HEIGHT));
      glutSetCursor(GLUT_CURSOR_NONE);
      glutFullScreen(); 
    }
    else
    {
      glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
      glutReshapeWindow(OldWidth, OldHeight);
    }
    break;
  case GLUT_KEY_PAGE_DOWN + 1000:
    glutPostRedisplay();
    break;
  }
}

void SpecialKeyPress(int key, int x, int y)
{
  LastMouseX = x;
  LastMouseY = y;
  Menu(key+1000, x, y);
}

// KeyPress name is already taken on linux
void KeyPress_(unsigned char key, int x, int y)
{
  LastMouseX = x;
  LastMouseY = y;
  Menu((int) key, x, y);
}

void MenuPress(int c)
{
  Menu(c, LastMouseX, LastMouseY);
}

void MenuStatus(int bob, int x, int y)
{
  LastMouseX = x;
  LastMouseY = y;
}

void Motion(int x, int y)
{
  LastMouseX = x;
  LastMouseY = y;
}

void Mouse(int button, int state, int x, int y)
{
  if(state) return;
}

// Load the environment geometry
void initModelForOpenGL(const std::string& filename, const optix::Matrix4x4& transform)
{
  std::cerr << "Loading " << filename << '\n';
  // For loading into OpenGL
  OGLModel = new nv::Model();
  if(!OGLModel->loadModelFromFile(filename.c_str())) {
    std::stringstream ss;
    ss << "loadModelFromFile('" << filename << "') failed" << std::endl;
    throw Exception(ss.str());
  }
  std::cerr << "loaded\n";

  OGLModel->removeDegeneratePrims();
  OGLModel->clearNormals();
  OGLModel->computeNormals();

  OGLModel->clearTexCoords();
  OGLModel->clearColors();
  OGLModel->clearTangents();

  OGLModel->compileModel();

  // Calculate bbox of model
  nv::vec3f modelBBMin, modelBBMax;
  OGLModel->computeBoundingBox(modelBBMin, modelBBMax);
  WorldAABB.include(make_float3(modelBBMin.x, modelBBMin.y, modelBBMin.z));
  WorldAABB.include(make_float3(modelBBMax.x, modelBBMax.y, modelBBMax.z));

  std::cerr << "WorldAABB=" << WorldAABB.m_min << WorldAABB.m_max << '\n';

  glGenBuffers(1, &modelVB);
  glBindBuffer(GL_ARRAY_BUFFER, modelVB);
  glBufferData(GL_ARRAY_BUFFER,
    OGLModel->getCompiledVertexCount() * OGLModel->getCompiledVertexSize() * sizeof(float),
    OGLModel->getCompiledVertices(), GL_STATIC_READ);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glGenBuffers(1, &modelIB);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, modelIB);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
    OGLModel->getCompiledIndexCount() * sizeof(int),
    OGLModel->getCompiledIndices(), GL_STATIC_READ);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

int initGLUT(int argc, char **argv)
{
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);    
  glutInitWindowSize(Width, Height);
  glutCreateWindow("Collision Detection");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(KeyPress_);
  glutSpecialFunc(SpecialKeyPress);
  glutMouseFunc(Mouse);
  glutMotionFunc(Motion);
  glutPassiveMotionFunc(Motion);
  glutIdleFunc(idle);

  glutMenuStatusFunc(MenuStatus);
  glutCreateMenu(MenuPress);
  glutAddMenuEntry("f: Full screen", 'f');
  glutAddMenuEntry("<esc> or q: exit program", '\033');
  glutAttachMenu(GLUT_RIGHT_BUTTON);

  bool GotExt = true;
  GotExt = (glewInit() == GLEW_OK) && GotExt;
  GotExt = GLEW_VERSION_1_4 && GotExt;
  GotExt = GLEW_NV_vertex_program && GotExt;
  GotExt = GLEW_NV_fragment_program && GotExt;

  if (!GotExt || !glewIsSupported("GL_VERSION_2_0 "
    "GL_EXT_framebuffer_object "))
  {
    std::cerr << "Unable to load the necessary extensions\n"
              << "This sample requires:\n"
              << "OpenGL 2.0\n"
              << "GL_EXT_framebuffer_object\n"
              << "Exiting...\n" << std::endl;
    exit(-1);
  }

  return 0;
}

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  -d  | --dim=<width>x<height>               Set image dimensions\n"
    << "  -p  | --pose=<cam>                         Initial camera string: \"[eye][lookat][up]vfov\", e.g. \"[0,0,-1][0,0,0][0,1,0]45.0\"\n"
    << "  -b  | --benchmark[=<w>x<t>]                Render and display 'w' warmup and 't' timing frames, then exit\n"
    << "  -B  | --benchmark-no-display=<w>x<t>       Render 'w' warmup and 't' timing frames, then exit\n"
    << "  -m  | --mode <N>                           Simulation mode: SIM_OBJECTS_AVOID_WALLS\n"
    << "  -n  | --num_objects <N>                    Number of moving objects to simulate\n"
    << "  -P  | --num_probes <N>                     For simulations that use collision probes, how many per moving object\n"
    << "  -v  | --vis_mat                            Visualize the collision matrices\n"
    << std::endl;

  std::cerr
    << "App keystrokes:\n"
    << "  v Toggle visualization of matrices\n"
    << "  < Decrease matrix visualization brightness scaling\n"
    << "  > Increase matrix visualization brightness scaling\n"
    << std::endl;

  if ( doExit ) exit(1);
}

void initialize(int argc, char **argv)
{
  size_t NObjects = 256u, NProbes = 64u;
  std::string objfilename;
  bool specifiedPose = false;
  bool dobenchmark = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      printUsageAndExit(argv[0]);
    } else if (arg == "--obj" || arg == "-o") {
      if (i == argc-1) {
        printUsageAndExit(argv[0]);
      }
      objfilename = argv[++i];
    } else if (arg == "--vis_mat" || arg == "-v") {
      VisualizeMatrices = true;
    } else if (arg == "--ao_hack" || arg == "-a") {
      if (i == argc-1) {
        printUsageAndExit(argv[0]);
      }
      AOHackfilename = argv[++i];
    } else if (arg.substr(0, 3) == "-p=" || arg.substr(0, 7) == "--pose=") {
      std::string camstr = arg.substr(arg.find_first_of('=') + 1);
      std::cerr << "Pose=" << camstr << std::endl;
      std::istringstream istr(camstr);
      istr >> Eye >> LookAt >> Up >> VFovDeg;
      specifiedPose = true;
    } else if (arg.substr(0, 3) == "-b=" || arg.substr(0, 23) == "--benchmark=") {
      dobenchmark = true;
      std::string bnd_args = arg.substr(arg.find_first_of('=') + 1);
      if (sutilParseImageDimensions(bnd_args.c_str(), &warmup_frames, &timed_frames) != RT_SUCCESS) {
        std::cerr << "Invalid --benchmark-no-display arguments: '" << bnd_args << "'" << std::endl;
        printUsageAndExit(argv[0]);
      }
    } else if (arg == "-b" || arg == "--benchmark") {
      dobenchmark = true;
    } else if (arg.substr(0, 3) == "-B=" || arg.substr(0, 23) == "--benchmark-no-display=") {
      benchmark_no_display = true;
      std::string bnd_args = arg.substr(arg.find_first_of('=') + 1);
      if (sutilParseImageDimensions(bnd_args.c_str(), &warmup_frames, &timed_frames) != RT_SUCCESS) {
        std::cerr << "Invalid --benchmark-no-display arguments: '" << bnd_args << "'" << std::endl;
        printUsageAndExit(argv[0]);
      }
    } else if (arg == "-B" || arg == "--benchmark-no-display") {
      benchmark_no_display = true;
    } else if( arg.substr( 0, 3) == "-d=" || arg.substr( 0, 6 ) == "--dim=" ) {
      std::string bnd_args = arg.substr(arg.find_first_of('=') + 1);
      if (sutilParseImageDimensions(bnd_args.c_str(), (unsigned int *)&Width, (unsigned int *)&Height) != RT_SUCCESS) {
        std::cerr << "Invalid --dim arguments: '" << bnd_args << "'" << std::endl;
        printUsageAndExit(argv[0]);
      }
    } else if (arg == "--mode" || arg == "-m") {
      if (i == argc-1) {
        printUsageAndExit(argv[0]);
      }
      simMode = static_cast<SimModes_E>(atoi(argv[++i]));
      if(simMode < 0 || simMode >= SIM_NUM_MODES)
        printUsageAndExit(argv[0]);
    } else if (arg == "--num_objects" || arg == "-n") {
      if (i == argc-1) {
        printUsageAndExit(argv[0]);
      }
      NObjects = atoi(argv[++i]);
    } else if (arg == "--num_probes" || arg == "-P") {
      if (i == argc-1) {
        printUsageAndExit(argv[0]);
      }
      NProbes = atoi(argv[++i]);
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit(argv[0]);
    }
  }
  if(objfilename.empty()) {
    objfilename = std::string(sutilSamplesDir()) + "/collision/Mazes/maze_32_1000walls_1_16904.obj";
    specifiedPose = true;
  }

  if(AOHackfilename.empty()) {
    std::string tmpfname = objfilename + ".ppm";
    std::ifstream tmpf(tmpfname.c_str());
    if(tmpf.is_open()) {
      AOHackfilename = tmpfname;
      tmpf.close();
    }
  }

  if( !dobenchmark ) printUsageAndExit(argv[0], false);

  if( NObjects < 1u ) printUsageAndExit(argv[0]);

  initGLUT(argc, argv);

  optix::Matrix4x4 ident;
  initModelForOpenGL(objfilename, ident);

  if(!specifiedPose) LookAt = WorldAABB.center();

  Scene = new CollisionScene(NObjects, NProbes, objfilename, WorldAABB, simMode, modelVB, modelIB, OGLModel);

  if(benchmark_no_display) {
    Benchmark(warmup_frames, timed_frames);
    exit(0);
  } else if(dobenchmark) {
    glutIdleFunc(idle);
  } else {
    warmup_frames = timed_frames = 0;
  }
}

int main(int argc, char **argv)
{
  try {
    initialize(argc, argv);

    glutMainLoop();
  } catch (Exception &E) {
    std::cerr << "Failure while initializing: " << E.getErrorString() << '\n';
    return 2;
  }
}

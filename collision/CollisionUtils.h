
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

// CollisionUtils.h
//
// Helper functions used by the collision detection code

#ifndef _collision_utils_h
#define _collision_utils_h

#include <optix_world.h>

#include <stdlib.h>

using namespace optix;

#define GL_ASSERT() {GLenum TMP_err; while ((TMP_err = glGetError()) != GL_NO_ERROR) \
    { std::ostringstream Er; Er << "OpenGL error: " << (char *)gluErrorString(TMP_err) << " at " << __FILE__ <<":" << __LINE__; throw Exception(Er.str()); } }

// A random number on 0.0 to 1.0
template<class Elem_T> inline Elem_T TRand()
{
#ifdef WIN32
    return static_cast<Elem_T>((rand()<<15)|rand()) / static_cast<Elem_T>(RAND_MAX*RAND_MAX);
#else
    return static_cast<Elem_T>(drand48());
#endif
}

template<class Elem_T> inline Elem_T TRand(const Elem_T low, const Elem_T high) { return low + TRand<Elem_T>() * (high - low); } // A random number on low to high
inline double DRand(const double low=0.0, const double high=1.0) { return low + TRand<double>() * (high - low); } // A random number on low to high
inline float DRandf(const float low=0.0f, const float high=1.0f) { return low + TRand<float>() * (high - low); } // A random number on low to high

inline float3 MakeDRand(const float low, const float high)
{
    return make_float3(TRand<float>(low, high), TRand<float>(low, high), TRand<float>(low, high));
}

inline float3 MakeDRand(const float3 low, const float3 high)
{
    return make_float3(TRand<float>(low.x, high.x), TRand<float>(low.y, high.y), TRand<float>(low.z, high.z));
}

// A random number
inline int LRand()
{
#ifdef WIN32
    return abs((rand() ^ (rand() << 15) ^ (rand() << 30)));
#else
    return int(lrand48());
#endif
}

inline int LRand(const int low, const int high) { return low + (LRand() % (high - low)); } // A random number on low to high

#endif

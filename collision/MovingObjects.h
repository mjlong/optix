
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

// MovingObjects.h
//
// A list of all the objects moving around in the scene. These must not collide with the environment.

#include "CollisionUtils.h"

#include <optix_world.h>

#include <vector>
// #include <iostream>

const float ObjSpeed = 0.1f;

class MovingObjects : public std::vector<float3>
{
public:
    MovingObjects() {}
    MovingObjects(size_t N, const optix::Aabb &aabb)
    {
        MakeRandomObjects(N, aabb);
    }

    void MakeRandomObjects(size_t N, const optix::Aabb &aabb)
    {
        for(size_t i=0; i<N; i++) {
            float3 C = MakeDRand(aabb.m_min, aabb.m_max);
            push_back(C);
            Vels.push_back(MakeDRand(-ObjSpeed, ObjSpeed));
        }
    }

    std::vector<float3> Vels; // The velocity vectors
    size_t CurSize; // The number that are being simulated on this time step.
};

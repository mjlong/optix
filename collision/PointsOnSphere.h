
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

// PointsOnSphere.h
//
// A list of all the directions in which to trace rays

#include "CollisionUtils.h"

#include <optix_world.h>

#include <vector>
#include <iostream>

class PointsOnSphere : public std::vector<float3>
{
public:
    PointsOnSphere() {}
    PointsOnSphere(size_t N, bool in_z_plane = false)
    {
        MakeRandomPoints(N, in_z_plane);
    }

    void MakeRandomPoints(size_t N, bool in_z_plane = false)
    {
        float maxdot = 0.01f;
        int tries = 0;

        do {
            float3 V = in_z_plane ? MakeRandPointOnCircle() : MakeRandPointOnSphere();
            bool worked = AttemptInsert(V, maxdot);
            if(!worked) {
                tries++;
                if(tries >= 500) {
                    tries = 0;
                    maxdot *= 1.03f;
                }
            }
        } while(size() < N);
    }

private:
    bool AttemptInsert(const float3 V, const float maxdot)
    {
        for(size_t i=0; i<size(); i++) {
            if(dot(V, operator[](i)) > maxdot) {
                return false; // It was too close
            }
        }

        push_back(V);
        return true;
    }

    // Return a point on a unit sphere
    float3 MakeRandPointOnSphere()
    {
        float3 RVec;
        do {
            RVec = MakeDRand(-1, 1);
        } while(dot(RVec,RVec) > 1 || dot(RVec,RVec) == 0);

        RVec = normalize(RVec);

        return RVec;
    }

    // Return a point on a unit circle
    float3 MakeRandPointOnCircle()
    {
        float3 RVec;
        do {
            RVec = MakeDRand(-1, 1);
            RVec.z = 0;
        } while(dot(RVec,RVec) > 1 || dot(RVec,RVec) == 0);

        RVec = normalize(RVec);

        return RVec;
    }
};

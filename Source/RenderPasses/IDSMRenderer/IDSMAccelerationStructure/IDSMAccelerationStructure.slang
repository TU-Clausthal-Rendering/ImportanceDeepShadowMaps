#pragma once
#include "../ImportanceMapHelpers/HeadIndexFromSampleDistribution.slang"
#include "../SharedShadowDefines.slang"

import Rendering.ShadowMaps.ShadowTestFunctions;
import Rendering.ShadowMaps.ShadowMapHelpers;
__exported import Scene.Raytracing;
__exported import Scene.Intersection;

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

#ifndef ACCEL_USE_RAY_INLINE
#define ACCEL_USE_RAY_INLINE 1
#endif

#ifndef ACCEL_USE_ONE_AABB_FOR_ALL_LIGHTS
#define ACCEL_USE_ONE_AABB_FOR_ALL_LIGHTS 1
#endif

#ifndef USE_COLOR_TRANSPARENCY
#define USE_COLOR_TRANSPARENCY 0
#endif

#if USE_COLOR_TRANSPARENCY
#define SHADOW_DATA_FORMAT float3
#else
#define SHADOW_DATA_FORMAT float
#endif

struct ASShadowRayData {
    SHADOW_DATA_FORMAT transparency;
    float sampleCount;

    __init() {
        this.transparency = 0.0;
        this.sampleCount = 0.0;
    }
}

struct AccelShadowAttribs {
    float2 pixel;
}

struct IDSMAccelerationStructure
{
    static const bool kAccelUseRayInline = ACCEL_USE_RAY_INLINE;       // TODO currently ignored and always assumed true
    static const uint kMaxAtomicWeight = IMPORTANCE_MAX_ATOMIC_WEIGHT;                      // Max value to be added on the atomic counter
    static const bool kUseOneAABBForAllLight = ACCEL_USE_ONE_AABB_FOR_ALL_LIGHTS;
    static const uint kMaskISMMultFactor = MASK_ISM_MULT_FACTOR; // Multiplication factor for the ISM used by the mask

    cbuffer SMCB
    {
        uint2 gSMSize;
        float gNear;
        float gFar;
        uint gMipCount;      // Count of mip in the sample distribution texture
        uint gMaxBufferSize; // Max size of the buffer
    }

    cbuffer ShadowVPs
    {
        float4x4 gShadowMapVP[COUNT_LIGHTS];
        float4x4 gStaggeredDirVP; //Only valid if there is a directional light
    }

    RaytracingAccelerationStructure gShadowAS;
    StructuredBuffer<AABB> gShadowAABBs[COUNT_LIGHTS];
    StructuredBuffer<SHADOW_DATA_FORMAT> gAccelShadowData[COUNT_LIGHTS];
    RWTexture2D<uint> gImportanceMap[COUNT_LIGHTS];
    Texture2D<uint> gSampleDistribution[COUNT_LIGHTS];
    Texture2DArray<float> gShadowMask;
    Buffer<float> gMaskShadowMap[COUNT_LIGHTS];

    SamplerState gLinearSampler;
    SamplerState gPointSampler;

    /** Shots an infinetly small ray in the IDSM Acceleration Structure to check if there is a node at the ray origin
     * \param[in/out] rayData. Contains the transparency and sample count
     * \param[in] lightIdx. Light Index
     * \param[in] ray. The infinetly small ray
    */
    void traceAccelShadowInline(inout ASShadowRayData rayData, uint lightIdx, const RayDesc ray) {
        RayQuery<RAY_FLAG_NONE> rayQuery;
        uint rayFlags = 0;
        uint instanceMask = kUseOneAABBForAllLight ? 0xFF : 1 << lightIdx; //Set the right bit for the AS instance mask
        //Trace the ray using a ray querey
        rayQuery.TraceRayInline(gShadowAS, rayFlags, instanceMask, ray);
        const uint aabbIdx = kUseOneAABBForAllLight ? 0 : lightIdx;
        //Ray query trace loop
        while (rayQuery.Proceed()) {
            //Fetch corresponding AABB
            AABB aabb = gShadowAABBs[aabbIdx][rayQuery.CandidatePrimitiveIndex()];

            // Check if hit is valid (Sometimes a hit is triggered even if it is outside of actual AABB)
            if (!aabb.contains(ray.Origin))
                continue;

            //Get Node data and store transparency
            SHADOW_DATA_FORMAT transparency = gAccelShadowData[aabbIdx][rayQuery.CandidatePrimitiveIndex()];
            //We take the average transparency to be sure, however, for IDSM no more than 1 node should be fetched per ray
            rayData.transparency += transparency;
            rayData.sampleCount += 1;
        }
    }

    /** Caculates the shadow value by evaluating the shadow acceleration structure
     * \param[in] index of the current light
     * \param[in] world position of the current hit
     * \param[in] is the light directional (ortho projection)
     * \param[in] Importance. Is used to weight the access texture. Should be in (0, 1]
     * \param[in] Mask Flag. Flag if the mask is used and the ISM should be sampled
     * \return transparency in [0,1]
     */
    float3 eval(uint lightIdx, float3 posW, bool isDirectional = false, float importance = 1.0, uint maskFlag = TRANSPARENCY_MASK_FLAG_NONE) {
        //Get shadow coordinates for the current (world) position
        float4 shadowPos = mul(gShadowMapVP[lightIdx], float4(posW, 1));
        if (!isDirectional) {
            shadowPos.xyz /= shadowPos.w;
            shadowPos.z = (shadowPos.w - gNear) / (gFar - gNear); // Linear distance
        }
        shadowPos.y *= -1;                                    // Flip
        shadowPos.xy = shadowPos.xy * 0.5 + 0.5;              // UV from [-1,1] to [0,1]

        bool validShadow = true;
        // Check if (world) position is inside the Shadow Map
        if (any(shadowPos.xy < 0.0) || any(shadowPos.xy > 1.0)) {
            if (!isDirectional) {
                return float3(-1.f);
            }
            validShadow = false;
        }

        //If mask is enabled, check if the bit is set
        bool validBackprojection = true;
        if ((maskFlag & TRANSPARENCY_MASK_FLAG_USE_MASK) != 0) {
            float maskVal = gShadowMask.SampleLevel(gLinearSampler, float3(shadowPos.xy, lightIdx), 0);
            validBackprojection = maskVal > TRANSPARENCY_MASK_COMPAIRE_VALUE; // 1.0/255.0
            validShadow &= validBackprojection;
        }

        //Get pixel coordinates for the current shadow position
        const float2 shadowPixel = shadowPos.xy * gSMSize;  //UV to pixel
        float dist = isDirectional ? shadowPos.z : shadowPos.w; //Distance in shadow map        
        uint2 uShadowPixel;
        if (validBackprojection) {
            if (isDirectional) {
                // For directional we use the staggered MVP for access
                float2 dirShadowPixel = mul(gStaggeredDirVP, float4(posW, 1)).xy;
                dirShadowPixel.y *= -1;                            // Flip
                dirShadowPixel.xy = dirShadowPixel.xy * 0.5 + 0.5; // UV from [-1,1] to [0,1]
                if (any(dirShadowPixel.xy < 0.0) || any(dirShadowPixel.xy > 1.0))
                    validBackprojection = false;
                uShadowPixel = uint2(dirShadowPixel * gSMSize);
            }
            else
                uShadowPixel = uint2(shadowPixel);
        }

        // Add Importance of the current sample to the Importance Map
        uint backprojectionWeight = clamp(uint(importance * kMaxAtomicWeight),1, kMaxAtomicWeight);
        if(validBackprojection)
            InterlockedAdd(gImportanceMap[lightIdx][uShadowPixel], backprojectionWeight);

        //If reprojection was not valid, dispatch a ray as fallback
        if (!validShadow)
            return float3(-1.0);

        //Get opaque shadow from ISM if the mask is enabled
        if ((maskFlag & TRANSPARENCY_MASK_FLAG_USE_SM) != 0) {
            //Get head buffer index. We only need y component here (x is for Linked List IDSM)
            int2 selectedIndex = getIndexFromSD(gSampleDistribution[lightIdx], shadowPos.xy, gMipCount - 2, kMaskISMMultFactor);               

            // Check if index is bigger than max size. Can happen if sample count increases drastically compaired to previous frame
            if (selectedIndex.y >= 0 && (selectedIndex.y < gMaxBufferSize * kMaskISMMultFactor)) {
                float maskSMDepth = gMaskShadowMap[lightIdx][selectedIndex.y];
                if (maskSMDepth <= dist)
                    return float3(0.0);
            }
        }

        // If only one AABB Buffer/BLAS is used for all lights, offset the pixel pos on the x-axis
        if (kUseOneAABBForAllLight)
            shadowPixel.x += gSMSize.x * lightIdx;

        //Prepare the infinetly small ray.
        ASShadowRayData rayData = ASShadowRayData();
        RayDesc ray;
        ray.Origin = float3(shadowPixel.x, shadowPixel.y, dist); //xy is shadow UV, z is depth.
        ray.Direction = float3(0, 0, -1);
        ray.TMin = 0;
        ray.TMax = 1e-10;

        traceAccelShadowInline(rayData, lightIdx, ray);

        SHADOW_DATA_FORMAT transparency = (rayData.sampleCount > 0) ? (rayData.transparency / rayData.sampleCount) : 1.0;

        return transparency;
    }
    
} gIDSMAccelerationStructure;

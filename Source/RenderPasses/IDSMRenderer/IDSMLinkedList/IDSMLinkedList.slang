#pragma once
#include "../ImportanceMapHelpers/HeadIndexFromSampleDistribution.slang"
#include "../SharedShadowDefines.slang"
#include "IDSMLinkedListData.slang"


import Rendering.ShadowMaps.ShadowTestFunctions;
import Rendering.ShadowMaps.ShadowMapHelpers;
__exported import Scene.Raytracing;
__exported import Scene.Intersection;

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

struct IDSMLinkedListShader
{
    static const uint kMaxAtomicWeight = IMPORTANCE_MAX_ATOMIC_WEIGHT; // Max value to be added on the atomic counter
    static const uint kMaskISMMultFactor = MASK_ISM_MULT_FACTOR; // Multiplication factor for the ISM used by the mask

    cbuffer SMCB
    {
        uint2 gSMSize;       //Shadow Map Size
        float gNear;         //Shadow Near
        float gFar;          //Shadow far
        uint gMipCount;      //Count of mip in the sample distribution texture
        uint gMaxBufferSize; //Max size of the buffer
        uint gISMMaxSize;   //ISM max size
    }

    cbuffer ShadowVPs
    {
        float4x4 gShadowMapVP[COUNT_LIGHTS];
        float4x4 gStaggeredDirVP; //Only valid if there is a directional light
    }

    StructuredBuffer<IDSMLinkedListNode> gLinkedListData[COUNT_LIGHTS];
    Texture2D<uint> gSampleDistribution[COUNT_LIGHTS];
    RWTexture2D<uint> gImportanceMap[COUNT_LIGHTS];
    Texture2DArray<float> gShadowMask;
    Buffer<float> gMaskShadowMap[COUNT_LIGHTS];

    SamplerState gLinearSampler;
    SamplerState gPointSampler;
    
    /** Caculates the shadow value by evaluating the shadow acceleration structure
     * \param[in] index of the current light
     * \param[in] world position of the current hit
     * \param[in] is the light directional (ortho projection)
     * \param[in] Importance. Is used to weight the access texture. Should be in (0, 1]
     * \return visibility in [0,1]
     */
    float3 eval(uint lightIdx, float3 posW, bool isDirectional = false, float importance = 1.0, uint maskFlags = TRANSPARENCY_MASK_FLAG_NONE) {
        float4 shadowPos = mul(gShadowMapVP[lightIdx], float4(posW, 1));

        if (!isDirectional) {
            shadowPos.xyz /= shadowPos.w;
            shadowPos.z = (shadowPos.w - gNear) / (gFar - gNear); // Linear distance
        }
        shadowPos.y *= -1;                                    // Flip
        shadowPos.xy = shadowPos.xy * 0.5 + 0.5;              // UV from [-1,1] to [0,1]

        bool validShadow = true;
        // Check if inside the SM
        if (any(shadowPos.xy < 0.0) || any(shadowPos.xy > 1.0)) {
            if (!isDirectional) {
                return -1.f;
            }
            validShadow = false;
        }

        // Get mask value
        bool validBackprojection = true;
        if ((maskFlags & TRANSPARENCY_MASK_FLAG_USE_MASK) != 0) {
            float maskVal = gShadowMask.SampleLevel(gLinearSampler, float3(shadowPos.xy, lightIdx), 0);
            validBackprojection = maskVal > TRANSPARENCY_MASK_COMPAIRE_VALUE;
            validShadow &= validBackprojection;
        }

        const float2 shadowPixel = shadowPos.xy * gSMSize; // floor((shadowPos.xy * gSMSize) - 0.5) ; //  //TODO strange fix for normal
        float dist = isDirectional ? shadowPos.z : shadowPos.w;

        // Add access
        uint2 uShadowPixel;
        if (validBackprojection) {
            if (isDirectional) {
                // For directional use the staggered MVP for access
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
        
        uint backprojectionWeight = clamp(uint(importance * kMaxAtomicWeight),1, kMaxAtomicWeight);
        if(validBackprojection)
            InterlockedAdd(gImportanceMap[lightIdx][uShadowPixel], backprojectionWeight);

        //Raytraced shadows as fallback
        if (!validShadow)
            return -1.f;

        float3 visibility = 1.0;

        //Get head buffer index for IDSM(x) and ISM(y)
        uint ismMultFactor = ((maskFlags & TRANSPARENCY_MASK_FLAG_USE_SM) != 0) ? kMaskISMMultFactor : 1u;
        int2 selectedIndex = getIndexFromSD(gSampleDistribution[lightIdx], shadowPos.xy, gMipCount - 2, ismMultFactor);
        //Check if index is invalid (unshadowed)
        if (selectedIndex.x < 0)
            return visibility;

        if ((maskFlags & TRANSPARENCY_MASK_FLAG_USE_SM) != 0) {
            if (selectedIndex.y < gISMMaxSize * kMaskISMMultFactor) {
                float maskSMDepth = gMaskShadowMap[lightIdx][selectedIndex.y];
                if (maskSMDepth <= dist)
                    return float3(0.0);
            }
        }

        IDSMLinkedListNode n = gLinkedListData[lightIdx][selectedIndex.x];
        // traverse visibility until dist is reached
        while (n.depth < dist) {
            visibility = n.visibility;
            if (n.next == 0 || n.next >= gMaxBufferSize) break;
            n = gLinkedListData[lightIdx][n.next];
        }
        
        return visibility;
    }
    
} gIDSMLinkedList;

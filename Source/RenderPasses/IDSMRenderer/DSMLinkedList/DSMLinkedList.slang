#pragma once
#include "DSMLinkedListData.slang"
#include "../SharedShadowDefines.slang"
#include "Utils/Math/MathConstants.slangh"

import Rendering.ShadowMaps.ShadowTestFunctions;
import Rendering.ShadowMaps.ShadowMapHelpers;

#ifndef COUNT_LIGHTS
#define COUNT_LIGHTS 1
#endif

#ifndef LINKED_LIST_PCF
#define LINKED_LIST_PCF 0
#endif

#ifndef USE_LINKED_LIST_ARRAY
#define USE_LINKED_LIST_ARRAY 0
#endif

struct DSMLinkedList
{

    //Constant Buffers
    cbuffer SMCB
    {
        uint2 gSMSize;
        float gNear;
        float gFar;
        uint gMaxBufferSize; // Max size of the buffer
    }

    cbuffer ShadowVPs
    {
        float4x4 gShadowMapVP[COUNT_LIGHTS];
    }

    StructuredBuffer<DSMLinkedListNode> gLinkedListData[COUNT_LIGHTS];
    StructuredBuffer<DSMLinkedListNeighborsNode> gLinkedListNeighbors[COUNT_LIGHTS];
    Texture2DArray<float> gMaskShadowMap;
    SamplerState gMaskSampler;

    //Fetches the 
    float3 getLinkedVisibility(DSMLinkedListNode n, float dist, uint lightIdx)
    {
        float3 visibility = 1.0;
        while (n.depth < dist) {
            visibility = n.visibility;
            if (n.next == 0 || n.next >= gMaxBufferSize) break;
            n = gLinkedListData[lightIdx][n.next];
        }

        return visibility;
    }

    /** Caculates the shadow value by evaluating the shadow acceleration structure
    * \param[in] index of the current light
    * \param[in] world position of the current hit
    * \param[in] is the light directional (ortho projection)
    * \param[in] Importance. Is used to weight the access texture. Should be in (0, 1]
    * \return visibility in [0,1]
    */
    float3 eval(uint lightIdx, float3 posW, bool isDirectional = false, uint maskFlag = TRANSPARENCY_MASK_FLAG_NONE) {
        float4 shadowPos = mul(gShadowMapVP[lightIdx], float4(posW, 1));

        if (!isDirectional) {
            shadowPos.xyz /= shadowPos.w;
            shadowPos.z = (shadowPos.w - gNear) / (gFar - gNear); // Linear distance
        }
        shadowPos.y *= -1;                       // Flip
        shadowPos.xy = shadowPos.xy * 0.5 + 0.5; // UV from [-1,1] to [0,1]

        bool validShadow = true;
        // Check if inside the SM
        if (any(shadowPos.xy < 0.0) || any(shadowPos.xy > 1.0)) {
            if (!isDirectional) {
                return -1.f;
            }
            validShadow = false;
        }

        #if LINKED_LIST_PCF
            float2 gatherUV;
            float4 gatherWeight = getGatherWeights(shadowPos.xy, gSMSize.x, gatherUV);
            float2 shadowPixel = gatherUV.xy * gSMSize - 0.5;
        #else
            float2 shadowPixel = floor(shadowPos.xy * gSMSize) + 0.5;
        #endif
        const uint2 uShadowPixel = uint2(floor(shadowPixel));
        float dist = isDirectional ? shadowPos.z : shadowPos.w;

        // Raytraced shadows as fallback
        if (!validShadow)
            return -1.f;

        // Use Opaque SM
        float visibilityOpaqueSM = 1.0;
        if ((maskFlag & TRANSPARENCY_MASK_FLAG_USE_SM) != 0) {
            #if LINKED_LIST_PCF
                visibilityOpaqueSM = 0.0;
                float4 shadowZ = gMaskShadowMap.Gather(gMaskSampler, float3(gatherUV.xy, lightIdx));
                [unroll]
                for (uint i = 0; i < 4; i++) {
                    if (dist <= shadowZ[i]) {
                        visibilityOpaqueSM += gatherWeight[i];
                    }
                }
                if (visibilityOpaqueSM <= 0)
                    return float3(0);
            #else
                float shadowZ = gMaskShadowMap[uint3(uShadowPixel, lightIdx)];
                if(dist > shadowZ)
                    return float3(0);
            #endif
        }

        float3 visibility = 1.0;
        uint index = uShadowPixel.y * gSMSize.x + uShadowPixel.x;

        // Check if index is bigger than max size. Can happen if sample count increases drastically compaired to previous frame
        uint maxRes = gSMSize.x * gSMSize.y;
        if (index >= maxRes)
            return visibility;

        DSMLinkedListNode n = gLinkedListData[lightIdx][index];
        // traverse visibility until dist is reached

        #if LINKED_LIST_PCF
            float3 gatherVis[4] = { float3(1), float3(1), float3(1), float3(1) };
            while (n.depth < dist) {
                gatherVis[3] = n.visibility;
                if (n.next == 0 || n.next >= gMaxBufferSize) break;
                uint nextIndex = n.next;
                n = gLinkedListData[lightIdx][n.next];
                if (n.depth < dist)
                    index = nextIndex;
            }
            DSMLinkedListNeighborsNode neighbor = gLinkedListNeighbors[lightIdx][index];
            uint zIndex = neighbor.right;
            uint xIndex = neighbor.bot;
            DSMLinkedListNode nZ = {};
            nZ.depth = FLT_MAX;
            nZ = gLinkedListData[lightIdx][zIndex];
            DSMLinkedListNode nX = {};
            nX.depth = FLT_MAX;
            nX = gLinkedListData[lightIdx][xIndex];
            gatherVis[2] = getLinkedVisibility(nZ, dist, lightIdx);
            gatherVis[0] = getLinkedVisibility(nX, dist, lightIdx);
            uint yIndex = gLinkedListNeighbors[lightIdx][zIndex].bot;
            DSMLinkedListNode nY = {};
            nY.depth = FLT_MAX;
            nY = gLinkedListData[lightIdx][yIndex];
            gatherVis[1] = getLinkedVisibility(nY, dist, lightIdx);
            visibility = gatherWeight.x * gatherVis[0] + gatherWeight.y * gatherVis[1] + gatherWeight.z * gatherVis[2] + gatherWeight.w * gatherVis[3];
        #else
            visibility = getLinkedVisibility(n, dist, lightIdx);
        #endif

        return min(visibility, visibilityOpaqueSM);
    }
    
} gDSMLinkedList;

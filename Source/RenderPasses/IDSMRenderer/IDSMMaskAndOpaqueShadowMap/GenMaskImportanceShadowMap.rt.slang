#include "../ImportanceMapHelpers/RaySampleFromSampleDistribution.slang"
#include "../SharedShadowDefines.slang"
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"


import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;

#ifndef USE_BLACKLIST
#define USE_BLACKLIST 0
#endif

cbuffer CB
{
    uint gLightIdx; // Current light index
    float3 gLightPos;
    bool gIsDirectional; // is directional
    uint gMipCount;
    uint2 gSMRes;
    uint gFrameCount;
    uint _pad;
    float4x4 gInvViewProjection; // Inverse view projection
    float4x4 gViewProjection;    // View Projection for current light
}

// Textures
Texture2DArray<float> gMask; // Non-Opaque object mask
Texture2D<uint> gSampleDistribution; //sample Distribution
RWBuffer<float> gShadowMap;           // Shadow map
RWBuffer<uint> gDispatchFeedbackBuffer;

SamplerState gMaskSampler;

static const bool kUseMask = USE_MASK; // USe mask to reject
static const uint kMultFactor = MASK_ISM_MULT_FACTOR;
//Soft Shadows
static const bool kUseRandomSoftShadows = USE_RANDOM_RANDOM_SOFT_SHADOWS;
static const float kRandomSoftShadowsPosRadius = RANDOM_SOFT_SHADOWS_POS_RADIUS; // For Point and Spot
static const float kRandomSoftShadowsDirSpread = RANDOM_SOFT_SHADOWS_DIR_SPREAD; // For Directional lights

struct RayData
{
    float depth;

    __init() {
        this.depth = FLT_MAX;
    }
}


/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[in] lod Method for computing texture level-of-detail.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

//Shaders
[shader("miss")]
void miss(inout RayData rayData)
{

}

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    //Alpha text is only used when the blacklist is used (else all non-opaque geometry is culled)
    #if USE_BLACKLIST
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        const uint materialID = gScene.getMaterialID(instanceID);
        let lod = ExplicitLodTextureSampler(0.f);

        if(gScene.materials.isCastShadow(materialID))
            IgnoreHit();

        if (gScene.materials.alphaTest(v, materialID, lod, MIN_OPACITY)) 
            IgnoreHit();
    #endif
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    VertexData v = gScene.getVertexData(triangleHit);

    float4 posH = mul(gViewProjection, float4(v.posW, 1.0));
    rayData.depth = gIsDirectional ? posH.z : posH.w; // Linear depth
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;

    // Check if current thread needs to dispatch a ray sample
    uint totalRaySamples = gSampleDistribution.Load(uint3(0, 0, gMipCount - 1)); // Total number of allowed samples
    totalRaySamples *= kMultFactor;
    uint idx1D = pixel.x + pixel.y * frameDim.x;                    // Create an 1D weight from 2D

    //For feedback buffer
    if (idx1D == 0)
        gDispatchFeedbackBuffer[gLightIdx] = uint(round(totalRaySamples));

    if (idx1D > totalRaySamples)
        return;

    // Sample a pixel location by traversing the SD. The pixel location is scaled to the base resolution of the SD
    float4 pixelExtend = float4(-0.5, 0.5, -0.5, 0.5); //Unused for Linked List
    float2 pixelSample = generateRayTracingSampleFromSD(gSampleDistribution, idx1D, gSMRes, gMipCount - 2, gFrameCount, pixelExtend, kMultFactor);  

    float2 relFramePos = pixelSample / gSMRes;
    float maskVal = 1.0;
    if(kUseMask)
        maskVal = gMask.SampleLevel(gMaskSampler, float3(relFramePos, gLightIdx), 0);

    relFramePos = relFramePos * 2.0 - 1.0; // From [0,1] to [-1,1]
    relFramePos.y *= -1.f;

    RayData rayData = RayData();

    bool valid = true;
    
    valid = maskVal > TRANSPARENCY_MASK_COMPAIRE_VALUE; // 1.0/255.0

    if (valid) {
        float3 lightPos = gLightPos;
        float2 rndOffset = float2(0);
        // Random offset for position
        if (kUseRandomSoftShadows) {
            SampleGenerator sg = SampleGenerator(DispatchRaysIndex().xy, gFrameCount);
            if (gIsDirectional) {
                rndOffset = sample_disk_concentric(sampleNext2D(sg));
                rndOffset *= 1.0 / gSMRes * kRandomSoftShadowsDirSpread;
            } else {
                float3 rndPosOffset = sample_sphere(sampleNext2D(sg));
                lightPos += rndPosOffset * kRandomSoftShadowsPosRadius;
            }
        }

        float4 posWPixelNear = mul(gInvViewProjection, float4(relFramePos + rndOffset, 0.0, 1.0));
        float4 posWPixelFar = mul(gInvViewProjection, float4(relFramePos, 1.0, 1.0));

        posWPixelNear.xyz /= posWPixelNear.w;
        posWPixelFar.xyz /= posWPixelFar.w;
        float3 rayDirection;
        if (gIsDirectional) {
            rayDirection = posWPixelFar.xyz - posWPixelNear.xyz;
        } else {
            rayDirection = posWPixelNear.xyz - lightPos;
        }

        RayDesc ray;
        if (gIsDirectional) {
            ray.TMin = 0;
            ray.TMax = length(rayDirection);
            ray.Origin = posWPixelNear.xyz;
            ray.Direction = rayDirection / ray.TMax;
        } else {
            ray.TMin = length(rayDirection);                 // Always start at depth 0
            ray.TMax = distance(lightPos, posWPixelFar.xyz); // Real far for the pixel
            ray.Origin = lightPos;
            ray.Direction = rayDirection / ray.TMin; // World direction
        }

        // Trace ray
        uint rayFlags = USE_BLACKLIST ? RAY_FLAG_NONE : RAY_FLAG_CULL_NON_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, kInstanceNoParticle /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
    }

    //uint insertIndex = idx1D + frameDim.x * frameDim.y * kMultFactor * gLightIdx;
    gShadowMap[idx1D] = rayData.depth;
}

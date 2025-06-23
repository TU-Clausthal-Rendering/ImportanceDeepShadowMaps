#include "../SharedShadowDefines.slang"
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"


import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Scene.Raytracing;
import Scene.Intersection;
//import Utils.Sampling.SampleGenerator;
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
    uint2 gSMRes;
    uint gFrameCount;
    uint _pad;
    float4x4 gInvViewProjection; // Inverse view projection
    float4x4 gViewProjection;    // View Projection for current light
}

// Textures
Texture2DArray<float> gMask; // Non-Opaque object mask
Texture2D<float> gSampleDistribution; //sample Distribution
RWTexture2D<float> gShadowMap;           // Shadow map
//Buffer<float2> gHaltonSamples;

SamplerState gMaskSampler;

//static const bool kUseHaltonSamplePattern = USE_HALTON_SAMPLE_PATTERN;
//static const uint kNumHaltonSamples = NUM_HALTON_SAMPLES;
static const bool kUseMask = USE_MASK; //use Mask to reject

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
    float2 pixelOffset = float2(0.5);

    const uint2 smRes = gSMRes;
    //SampleGenerator sg = SampleGenerator(pixel, gFrameCount + 34 * gLightIdx);

    /*
    if (kUseHaltonSamplePattern) {
        uint jitterIdx = (subpixelLoc.x + (subpixelLoc.y * samplesSq) + gFrameCount) % kNumHaltonSamples;
        pixelOffset = gHaltonSamples[jitterIdx]; // pixel offset in [0,1]
    }
    /*
    else {
        pixelOffset = sampleNext2D(sg);
    }
    */

    //
    // Dispatch Sample / Create Shadow Map
    //

    float2 pixelSample = pixel + pixelOffset;

    float2 relFramePos = pixelSample / smRes;
    float maskVal = 1.0;
    if(kUseMask)
        maskVal = gMask.SampleLevel(gMaskSampler, float3(relFramePos, gLightIdx), 0);

    relFramePos = relFramePos * 2.0 - 1.0; // From [0,1] to [-1,1]
    relFramePos.y *= -1.f;

    RayData rayData = RayData();

    bool valid = true;
    
    valid = maskVal > TRANSPARENCY_MASK_COMPAIRE_VALUE; // 1.0/255.0

    if (valid) {
        //TODO Soft shadows

        float4 posWPixelNear = mul(gInvViewProjection, float4(relFramePos, 0.0, 1.0));
        float4 posWPixelFar = mul(gInvViewProjection, float4(relFramePos, 1.0, 1.0));

        posWPixelNear.xyz /= posWPixelNear.w;
        posWPixelFar.xyz /= posWPixelFar.w;
        float3 rayDirection;
        if (gIsDirectional) {
            rayDirection = posWPixelFar.xyz - posWPixelNear.xyz;
        } else {
            rayDirection = posWPixelNear.xyz - gLightPos;
        }

        RayDesc ray;
        if (gIsDirectional) {
            ray.TMin = 0;
            ray.TMax = length(rayDirection);
            ray.Origin = posWPixelNear.xyz;
            ray.Direction = rayDirection / ray.TMax;
        } else {
            ray.TMin = length(rayDirection);                 // Always start at depth 0
            ray.TMax = distance(gLightPos, posWPixelFar.xyz); // Real far for the pixel
            ray.Origin = gLightPos;
            ray.Direction = rayDirection / ray.TMin; // World direction
        }

        // Trace ray
        uint rayFlags = USE_BLACKLIST ? RAY_FLAG_NONE : RAY_FLAG_CULL_NON_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, kInstanceNoParticle /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
    }

    gShadowMap[pixel] = rayData.depth;
}

#include "../SharedShadowDefines.slang"
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"


import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;

#ifndef NUM_LIGHTS
#define NUM_LIGHTS 1
#endif

//Maximum index 
#ifndef MAX_IDX
#define MAX_IDX 1
#endif

#ifndef TRACE_NON_OPAQUE_ONLY
#define TRACE_NON_OPAQUE_ONLY 0
#endif

#ifndef USE_COLOR_TRANSPARENCY
#define USE_COLOR_TRANSPARENCY 0
#endif

#if USE_COLOR_TRANSPARENCY
#define SHADOW_DATA_FORMAT float3
#else
#define SHADOW_DATA_FORMAT float
#endif

/** Payload for shadow ray. (32B)
 */
struct RayData
{
    float3 currentTransmittance;
    float dist;
    float3 sampleTransmittance;
    float depth;

    __init() {
        this.dist = -1.0;
        this.depth = 1.0;
        this.currentTransmittance = float3(1.0);
        this.sampleTransmittance = float3(1.0);
    }

    const bool isOpaque() {
        return sampleTransmittance.x < 0;
    }
};

cbuffer CB
{
    uint gFrameCount;    // Current frame count
    float3 gLightPos;    // Current Light Position
    bool gIsDirectional; // Is light a directional light (posW invalid)
    float3 gLightDir;    // LightDirection
    float gFar;
    uint gLightIdx; // LightIndex
    uint2 gSMRes;   // ShadowMap resolution
    float gSpreadAngle;
    uint _pad;
    float4x4 gViewProj;
    float4x4 gInvViewProj; // Inverse view projection
}

RWStructuredBuffer<AABB> gAABB;
RWBuffer<uint> gCounter;
RWStructuredBuffer<SHADOW_DATA_FORMAT> gData;
Buffer<float2> gHaltonSamples;

static const float kMidpointPercentage = MIDPOINT_PERCENTAGE; // Percentage of distance the midpoint is set
static const float kMidpointDepthBias = MIDPOINT_DEPTH_BIAS;  // Depth bias for midpoint
static const bool kUseOneAABBForAllLights = USE_ONE_AABB_BUFFER_FOR_ALL_LIGHTS; //Use one buffer for all lights
static const float kMinOpacity = MIN_OPACITY;
static const float kMaxTransmittance = 1.0 - kMinOpacity;
static const bool kUseHaltonSamplePattern = USE_HALTON_SAMPLE_PATTERN;
static const uint kNumHaltonSamples = NUM_HALTON_SAMPLES;
static const float kPixelOffset = ACCEL_BOXES_PIXEL_OFFSET; // Either 0.5(Normal) or 1.0(PCF)
static const bool kUseMidpointBias = true; // Uses min between midpoint and a high bias (0.08% of lights far)
static const bool kIncludeCastShadowInstanceMaskBit = INCLUDE_CAST_SHADOW_INSTANCE_MASK_BIT;


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

// Get next hightest float
float ulp(float f)
{
    int i = asint(f);
    return abs(asfloat(i + 1) - f);
}

inline float getMidpointDepth(float lastHitDepth, float distanceDifference) {
    float midpointDepth = 0;
    if (kUseMidpointBias)
        midpointDepth = lastHitDepth + min(distanceDifference * kMidpointPercentage, kMidpointDepthBias);
    else
        midpointDepth = lastHitDepth + distanceDifference * kMidpointPercentage;

    return midpointDepth;
}

//Shaders
[shader("miss")]
void miss(inout RayData rayData)
{
    rayData.dist = -1.0;
}

//
// Shader entry points for TriangleMesh hit groups.
//

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    //  Alpha test for non-opaque geometry.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    VertexData v = gScene.getVertexData(triangleHit);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
     ITextureSampler lod = ExplicitLodTextureSampler(0.f);

    float3 transmittance = float3(0.0);

    if (false) { //Colored transmittance
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -WorldRayDirection(), lod);
        // uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        // Emissive lights
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        float3 color = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo;
        // Normalize color
        float maxColor = max(max(color.x, color.y), color.z);
        color /= maxColor;
        if (sd.opacity < kMinOpacity)
            transmittance = 1.0;
        else
            transmittance = color * (1.0 - sd.opacity);
    } else {
        transmittance = 1.0 - float3(gScene.materials.getMaterialOpacity(v, materialID, lod));
    }

    // If hit is transparent (alpha < 1e-7)
    if (all(transmittance > kMaxTransmittance))
        IgnoreHit();

    // Commit hit
    if (RayTCurrent() < rayData.dist) {
        rayData.sampleTransmittance = transmittance;
        rayData.dist = RayTCurrent();
    }
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    VertexData v = gScene.getVertexData(triangleHit);

    float3 transmittance = rayData.sampleTransmittance;
    rayData.sampleTransmittance = float3(1.0);
    // Case opaque object that does not call the anyHit
    if (RayTCurrent() < rayData.dist) {
        transmittance = 0.0;
        rayData.sampleTransmittance.x = -1.0;
    }

    rayData.currentTransmittance = rayData.currentTransmittance * transmittance;
    float4 posH = mul(gViewProj, float4(v.posW, 1.0));
    rayData.depth = gIsDirectional ? posH.z : posH.w; // Linear depth
    rayData.dist = RayTCurrent();
}

inline uint getInsertIndex() {
    uint insertIdx = 0;
    const uint lightIdx = kUseOneAABBForAllLights ? 0 : gLightIdx;
    InterlockedAdd(gCounter[lightIdx], 1u, insertIdx);
    return insertIdx;
}

// Writes the shadow map sample. Returns a bool if the insert was sucessfull
inline bool writeSample(float2 pixelSample, float4 boxPixelOffset, float boxMinDepth, float boxMaxDepth, float3 transmittance, float hitDepth) {
    float3 minPoint = float3(pixelSample.x + boxPixelOffset.x, pixelSample.y + boxPixelOffset.z, boxMinDepth);
    float3 maxPoint = float3(pixelSample.x + boxPixelOffset.y, pixelSample.y + boxPixelOffset.w, boxMaxDepth);

    AABB aabb = AABB(minPoint, maxPoint);
    uint insertIdx = getInsertIndex();
    if (insertIdx > MAX_IDX)
        return false; // Cannot append box

    gAABB[insertIdx] = aabb;
#if USE_COLOR_TRANSPARENCY
    gData[insertIdx] = transmittance;
#else
    gData[insertIdx] = transmittance.x;
#endif
    return true;
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    float4 boxPixelOffset = float4(-kPixelOffset, kPixelOffset, -kPixelOffset, kPixelOffset); // Pixel offset to create AABB (xMin, xMax, yMin, yMax)
    float2 pixelOffset = float2(0.5);
    if (kUseHaltonSamplePattern) {
        uint jitterIdx = (pixel.x + (pixel.y * frameDim.x) + gFrameCount) % kNumHaltonSamples;
        pixelOffset = gHaltonSamples[jitterIdx]; // pixel offset in [0,1]
    }
    float2 pixelSample = pixel + pixelOffset;

    float2 relFramePos = pixelSample / frameDim;
    relFramePos = relFramePos * 2.0 - 1.0; // From [0,1] to [-1,1]
    relFramePos.y *= -1.f;

    float4 posWPixelNear = mul(gInvViewProj, float4(relFramePos, 0.0, 1.0));
    float4 posWPixelFar = mul(gInvViewProj, float4(relFramePos, 1.0, 1.0));
    posWPixelNear.xyz /= posWPixelNear.w;
    posWPixelFar.xyz /= posWPixelFar.w;
    float3 rayDirection;
    if (gIsDirectional) {
        rayDirection = posWPixelFar.xyz - posWPixelNear.xyz;
    } else {
        rayDirection = posWPixelNear.xyz - gLightPos;
    }

    // Add offset to the pixel position if one aabb buffer is used
    // Currently this is solved simple by just adding an offset of the SM size * LightIndex in x direction
    if (kUseOneAABBForAllLights) {
        pixelSample.x += frameDim.x * gLightIdx;
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

    RayData rayData = RayData();
    bool rayTracingLoop = true;
    float aabbFrontZ = 0.0;
    float aabbBackZ = 0.0;
    float lastHitDepth = 0.0;
    float3 currTranmittance = float3(1.0);  // Tranmission end of current sample
    float3 prevTransmittance = float3(1.0); // Tranmission end of previous sample
    const float far = ray.TMax;
    const uint instanceMask = getBestParticleShadowPlane(gLightDir, kIncludeCastShadowInstanceMaskBit);
    bool currHitOpaque = false;
    bool prevHitOpaque = false;
    // Loop to get all possible samples
    for (uint i = 0; (i < MAX_ELEMENTS) && rayTracingLoop; i++) {
        rayData.depth = -1;
        rayData.dist = ray.TMax;
        rayData.sampleTransmittance = 0.0;

        // Trace ray
        uint rayFlags = 0;
        #if TRACE_NON_OPAQUE_ONLY
            rayFlags = RAY_FLAG_CULL_OPAQUE;
        #endif
        TraceRay(gScene.rtAccel, rayFlags /*Flags*/, instanceMask /* kShadowRayInstanceMask instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        // A miss on the first hit ends the ray
        if (i == 0 && rayData.dist < 0)
            return;

        // End Loop for the next iteration if transmittance is smaller as minValue or the ray missed the scene
        if (all(prevTransmittance <= kMinOpacity) || rayData.dist < 0) {
            rayTracingLoop = false;
        }

        // Get difference between last and current depth
        float distDiff = rayData.depth - lastHitDepth;
        // Update aabb Z values if the ray hit something
        if (rayData.dist >= 0) {
            aabbFrontZ = aabbBackZ;
            aabbBackZ = getMidpointDepth(lastHitDepth, distDiff);
        }

        // Write the sample stating on 3rd iteration(i=2). Both aabbZ values are filled at that point
        if (rayData.dist >= 0 && i > 1) {
            if (!writeSample(pixelSample, boxPixelOffset, aabbFrontZ, aabbBackZ, prevTransmittance, lastHitDepth))
                return;
        }

        // Update depth, transmittance and ray for next iteration
        if (rayData.dist >= 0) {
            prevHitOpaque = currHitOpaque;
            currHitOpaque = rayData.isOpaque();
            float rayOffset = ulp(rayData.dist) * 4.0; // Next highest float x4 seems to work pretty well
            // ray.TMin = currHitOpaque ? min(rayData.dist + kOpaqueHitRayDepthBias * pixelScale, rayData.dist + rayOffset) : rayData.dist + rayOffset; // This a bit of a dirty fix for particles that overlap with solid geometry
            ray.TMin = rayData.dist + rayOffset;
            lastHitDepth = rayData.depth;
            prevTransmittance = currTranmittance;
            currTranmittance = rayData.currentTransmittance;
        }
    }

    // Insert last two elements
    bool lastHitMiss = rayData.dist < 0;
    bool storePrev = true;
    bool storeCurr = !lastHitMiss;
    #if TRACE_NON_OPAQUE_ONLY
        storeCurr = true; // It now cannot be assumed that this is the last hit
    #endif

    // Update box
    aabbFrontZ = aabbBackZ;
    #if TRACE_NON_OPAQUE_ONLY 
        aabbBackZ = lastHitDepth + kMidpointDepthBias; //No midpoint for this hit
    #else
        aabbBackZ = lastHitMiss || !storeCurr ? far : getMidpointDepth(lastHitDepth, (rayData.depth - lastHitDepth));
    #endif

    if (any(prevTransmittance < 1.0) && storePrev) {
        if (!writeSample(pixelSample, boxPixelOffset, aabbFrontZ, aabbBackZ, prevTransmittance, lastHitDepth))
            return;
    }
    // Store last box
    if (any(currTranmittance < 1.0) && storeCurr) {
        if (!writeSample(pixelSample, boxPixelOffset, aabbBackZ, far, currTranmittance, rayData.depth))
            return;
    }
}

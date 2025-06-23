#include "../ImportanceMapHelpers/RaySampleFromSampleDistribution.slang"
#include "../SharedShadowDefines.slang"
#include "IDSMLinkedListData.slang"
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"


import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;

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
    uint gFrameCount; // Current frame count
    float3 gLightPos; // Current Light Position
    bool gIsDirectional; // Is light a directional light (posW invalid)
    float3 gLightDir;   //Light Direction
    float gFar;
    uint gLightIdx;     //LightIndex
    uint gMipCount;     // Number of mips
    uint2 gSMRes;       // ShadowMap resolution
    float gSpreadAngle;
    uint _pad;
    float4x4 gViewProj;
    float4x4 gInvViewProj; // Inverse view projection
}

RWBuffer<uint> gCounter;
RWStructuredBuffer<IDSMLinkedListNode> gData;
RWTexture2D<uint> gImportanceMap;
Texture2D<uint> gSampleDistribution;

static const float kMidpointPercentage = MIDPOINT_PERCENTAGE; // Percentage of distance the midpoint is set
static const float kMidpointDepthBias = MIDPOINT_DEPTH_BIAS; // Depth bias for midpoint
static const float kMinOpacity = MIN_OPACITY;
static const float kMaxTransmittance = 1.0 - kMinOpacity;
static const float kPixelOffset = ACCEL_BOXES_PIXEL_OFFSET; // Either 0.5(Normal) or 1.0(PCF)
static const uint kMipLevels = SAMPLE_DIST_MIPS;
static const bool kUseRandomSoftShadows = USE_RANDOM_RANDOM_SOFT_SHADOWS;
static const float kRandomSoftShadowsPosRadius = RANDOM_SOFT_SHADOWS_POS_RADIUS; // For Point and Spot
static const float kRandomSoftShadowsDirSpread = RANDOM_SOFT_SHADOWS_DIR_SPREAD; // For Directional lights
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

    if (!gScene.materials.isAlphaTestSupported(materialID))
        return;

    ITextureSampler lod = ExplicitLodTextureSampler(0.f);
    float3 transmittance = float3(0.0);

    if(kUseColorTransparency){
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -WorldRayDirection(), lod);
        //uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        // Emissive lights
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        float3 color = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo;
        //Normalize color
        float maxColor = max(max(color.x, color.y), color.z);
        color /= maxColor;
        if(sd.opacity < kMinOpacity)
            transmittance = 1.0;
        else
            transmittance = color * (1.0 - sd.opacity);
    }else{
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
    //Case opaque object that does not call the anyHit
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
    const uint lightIdx = gLightIdx;
    InterlockedAdd(gCounter[lightIdx], 1u, insertIdx);
    return insertIdx;
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;

    // Clear access counter
    if (all(pixel < gSMRes)) {
        gImportanceMap[pixel] = 0; // clear
    }

    // Check if current thread needs to dispatch a ray sample
    uint totalRaySamples = gSampleDistribution.Load(uint3(0, 0, gMipCount - 1)); // Total number of allowed samples
    uint idx1D = pixel.x + pixel.y * frameDim.x;                    // Create an 1D weight from 2D
    if (idx1D > totalRaySamples)
        return;

    // Sample a pixel location by traversing the SD. The pixel location is scaled to the base resolution of the SD
    float4 pixelExtend = float4(-0.5, 0.5, -0.5, 0.5); // Unused for Linked List
    float2 pixelSample = generateRayTracingSampleFromSD(gSampleDistribution, idx1D, gSMRes, gMipCount - 2, gFrameCount, pixelExtend);

    float2 relFramePos = pixelSample / gSMRes;
    relFramePos = relFramePos * 2.0 - 1.0; // From [0,1] to [-1,1]
    relFramePos.y *= -1.f;

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

    float4 posWPixelNear = mul(gInvViewProj, float4(relFramePos + rndOffset, 0.0, 1.0));
    float4 posWPixelFar = mul(gInvViewProj, float4(relFramePos, 1.0, 1.0));

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

    RayData rayData = RayData();
    bool rayTracingLoop = true;
    float midpointLast = 0.0;
    float midpointCurrent = 0.0;
    float lastHitDepth = 0.0;
    float3 currTranmittance = float3(1.0);  //Tranmission end of current sample
    float3 prevTransmittance = float3(1.0); //Tranmission end of previous sample
    const float far = ray.TMax;

    IDSMLinkedListNode prevNode;
    prevNode.depth = FLT_MAX;
    prevNode.visibility = 1.0;
    prevNode.next = 0;
    uint prevNodeIndex = idx1D;
    uint nodeIndex = prevNodeIndex;
    const uint instanceMask = getBestParticleShadowPlane(gLightDir,kIncludeCastShadowInstanceMaskBit);
    bool currHitOpaque = false;
    bool prevHitOpaque = false;

    // Loop to get all possible samples
    for (uint i = 0; (i < MAX_ELEMENTS) && rayTracingLoop; i++) {
        rayData.depth = -1;
        rayData.dist = ray.TMax;
        rayData.sampleTransmittance = 1.0;

        // Trace ray
        uint rayFlags = 0;
    #if TRACE_NON_OPAQUE_ONLY
        rayFlags = RAY_FLAG_CULL_OPAQUE;
    #endif
        TraceRay(gScene.rtAccel, rayFlags /*Ray flags*/, instanceMask /* kShadowRayInstanceMask instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);

        // A miss on the first hit ends the ray
        if (i == 0 && rayData.dist < 0) {
            break;
        }

        // End Loop for the next iteration if transmittance is smaller as minValue or the ray missed the scene
        if (all(prevTransmittance <= kMinOpacity) || rayData.dist < 0) {
            rayTracingLoop = false;
        }

        // Get difference between last and current depth
        float distDiff = rayData.depth - lastHitDepth;
        // Update midpoint depth values if the ray hit something
        if (rayData.dist >= 0) {
            midpointLast = midpointCurrent;
            midpointCurrent = getMidpointDepth(lastHitDepth, distDiff);
        }

        // Write the sample stating on 3rd iteration(i=2).
        if (rayData.dist >= 0 && i > 1) {
            nodeIndex = getInsertIndex();
            if (nodeIndex > MAX_IDX)
                rayTracingLoop = false;

            prevNode.next = nodeIndex;
            gData[prevNodeIndex] = prevNode;
        }

        // Update depth, transmittance and ray for next iteration
        if (rayData.dist >= 0) {
            prevHitOpaque = currHitOpaque;
            currHitOpaque = rayData.isOpaque();
            float rayOffset = ulp(rayData.dist) * 4.0; // Next highest float x4 seems to work pretty well
            ray.TMin = rayData.dist + rayOffset;
            lastHitDepth = rayData.depth;
            prevTransmittance = currTranmittance;
            currTranmittance = rayData.currentTransmittance;

            // Update node
            prevNode.depth = midpointCurrent;
            #if USE_COLOR_TRANSPARENCY
                prevNode.visibility = prevTransmittance;
            #else
                prevNode.visibility = prevTransmittance.x;
            #endif
            prevNode.next = 0;
            prevNodeIndex = nodeIndex;
        }
    }

    // Insert last two elements
    bool storeLastBox = true;
#if TRACE_NON_OPAQUE_ONLY
    storeLastBox &= any(currTranmittance < 1.0);
    midpointLast = lastHitDepth + kMidpointDepthBias; //No midpoint for this hit
#else
    storeLastBox &= (any(currTranmittance < 1.0) && rayData.dist >= 0);
    midpointLast = midpointCurrent;
#endif
    storeLastBox |= (prevNodeIndex == idx1D) && all(prevTransmittance >= 1.0); //Guarantees that the head buffer is set

#if TRACE_NON_OPAQUE_ONLY
    if (any(prevTransmittance < 1.0) || storeLastBox)
#else
    if (any(prevTransmittance < 1.0))
#endif
    {
        if (storeLastBox){
            nodeIndex = getInsertIndex();
            if(nodeIndex > MAX_IDX)
                return;
        }

        prevNode.next = storeLastBox ? nodeIndex : 0;
        gData[prevNodeIndex] = prevNode;
        prevNode.depth = midpointLast;
        #if USE_COLOR_TRANSPARENCY
            prevNode.visibility = currTranmittance;
        #else
            prevNode.visibility = currTranmittance.x;
        #endif
        prevNode.next = 0;
        prevNodeIndex = nodeIndex;
    }

    // Store last box
    if (storeLastBox) {
        gData[prevNodeIndex] = prevNode;
    }
}

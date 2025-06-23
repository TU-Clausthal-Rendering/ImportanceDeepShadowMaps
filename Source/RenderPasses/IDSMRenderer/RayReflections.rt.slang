#include "EvaluateAnalyticLight.slang"

import Utils.Math.MathHelpers;
import Rendering.Materials.Fresnel;

cbuffer CB
{
    uint gFrameCount; // Current it for sample gen
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gThp;
Texture2D<float> gReflectionMask;

// Outputs
RWTexture2D<float4> gOutputColor;

#define is_valid(name) (is_valid_##name != 0)
static const float kMinOpacity = MIN_OPACITY;
static const uint kMaxElements = MAX_ELEMENTS; // For now 256 max transparent elements on a ray
static const bool kAdjustShadingNormals = true;
static const float kAmbientStrength = AMBIENT_STRENGTH;
static const float kEnvMapStrength = ENV_MAP_STRENGTH;
static const bool kUseIrregularShadowMask = USE_IRRGEGULAR_SHADOW_MASK;


/** Payload for alpha ray (20B + 16B if lod is enabled)
*/
struct TransparencyRayData {
    PackedHitInfo hit; //< Packed Hit
    float dist;
    bool isOpaque;

    __init(float tMax)
    {
        this.hit = {}; // Init emtpy
        this.dist = tMax;
        this.isOpaque = false;
    }
}

/** Struct with data for the current path
*/
struct PathData
{
    float3  radiance;       ///< Accumulated outgoing radiance from path.
    bool    terminated;     ///< Set to true when path is terminated.
    float3  thp;            ///< Current path throughput. This is updated at each path vertex.
    uint    pathLength;     ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;          ///< Next path segment origin.
    float rayTransparency;  ///< Full transparency for the ray
    float3 direction;       ///< Next path segment direction.
    float alphaDist;        ///< Distance when the alpha ray should start

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Initializes ray payload with default parameters.
    */
    __init()
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.alphaDist = 0.0;
    }
};

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[in] lod Method for computing texture level-of-detail.
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod, inout float minTriangleSide)
{
    StaticVertexData vertices[3];
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit, vertices);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    // Calc min triangle side
    minTriangleSide = min( min(distance(vertices[0].position, vertices[1].position), distance(vertices[1].position, vertices[2].position)), distance(vertices[0].position, vertices[2].position));

    return sd;
}

[shader("miss")]
void miss(inout TransparencyRayData rayData)
{
    rayData.dist = -1.0f; 
}

[shader("anyhit")]
void anyHit(inout TransparencyRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
     // Alpha test for non-opaque geometry.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ITextureSampler lod = ExplicitLodTextureSampler(0.f);

    if (!gScene.materials.alphaTest(v, materialID, lod, kMinOpacity)) {
        //Update payload
        if (RayTCurrent() < rayData.dist) {
           
            rayData.hit = HitInfo(triangleHit).getData();
            rayData.dist = RayTCurrent();
            rayData.isOpaque = kShadowMaterialFlagAsBlacklist ? !gScene.materials.isCastShadow(materialID) : false;
        }
    } else {
        IgnoreHit();
    }
}

[shader("closesthit")]
void closestHit(inout TransparencyRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    if (RayTCurrent() < rayData.dist) {
        // Alpha test for non-opaque geometry.
        TriangleHit triangleHit;
        triangleHit.instanceID = getGeometryInstanceID();
        triangleHit.primitiveIndex = PrimitiveIndex();
        triangleHit.barycentrics = attribs.barycentrics;

        VertexData v = gScene.getVertexData(triangleHit);
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        ITextureSampler lod = ExplicitLodTextureSampler(0.f);

        rayData.hit = HitInfo(triangleHit).getData();
        rayData.dist = RayTCurrent();
        rayData.isOpaque = true;
    }
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 frameDim = DispatchRaysDimensions().xy;

    //Check if this is a reflective surface
    if (gReflectionMask[pixel] <= 0)
        return;

    // Load Hit and View vector
    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    float3 viewDir;
    viewDir = gScene.camera.computeRayPinhole(pixel, frameDim).dir;

    // Load shading data.
    ITextureSampler lod = ExplicitLodTextureSampler(0.f);
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -viewDir, lod);
    uint hints = 0;
    if (kAdjustShadingNormals)
        hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    BSDFProperties bsdfProperties = mi.getProperties(sd);

    // Prepare ray payload.
    PathData pd = PathData();
    pd.origin = gScene.camera.getPosition();
    pd.direction = reflect(viewDir, bsdfProperties.guideNormal);
    pd.sg = SampleGenerator(pixel, gFrameCount + 32);

    //Get Reflection throughput
    //pd.thp = gThp[pixel].xyz * (mi.eval(sd, pd.direction, pd.sg)/  mi.evalPdf(sd, pd.direction)); //Same result als below, but much more expensive
    pd.thp = gThp[pixel].xyz * evalFresnelSchlick(bsdfProperties.specularReflectionAlbedo, 1.0f, max(0.00001f, dot(bsdfProperties.guideNormal, pd.direction))); 
    
    //Prepare Ray
    RayDesc ray;
    ray.Origin = sd.computeNewRayOrigin();
    ray.Direction = pd.direction;
    ray.TMin = 0;
    ray.TMax = gScene.camera.data.farZ; ;
    TransparencyRayData rayData = TransparencyRayData(ray.TMax);
    const uint instanceMask = getBestParticleShadowPlane(ray.Direction,true);

    //Main transparency loop
    for(uint i=0; i<kMaxElements && !pd.terminated; i++){
        // Trace Ray
        uint rayFlags = 0;
        TraceRay(gScene.rtAccel, rayFlags, instanceMask , 0, rayTypeCount, 0, ray, rayData);
        HitInfo hit = HitInfo(rayData.hit);

        // Miss, evaluate Env Map
        if (rayData.dist < 0 || !hit.isValid()){
            pd.terminated = true;
            pd.radiance += gScene.envMap.eval(viewDir) * kEnvMapStrength * pd.thp;
            break;
        }

        ITextureSampler lod = ExplicitLodTextureSampler(0.f);
        // Get shading data
        float minTriangleSide = 0.0;
        ShadingData sd = loadShadingData(hit,pd.direction, lod, minTriangleSide);
        // Get material instance
        uint hints = 0;
        if (kAdjustShadingNormals)
            hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);

        //Emissive light hits
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        pd.radiance += pd.thp * bsdfProperties.emission * sd.opacity;

        // Analytic Lights
        uint maskFlag = TRANSPARENCY_MASK_FLAG_NONE;
        if(kUseIrregularShadowMask)
            maskFlag = rayData.isOpaque ? TRANSPARENCY_MASK_FLAG_USE_MASK : TRANSPARENCY_MASK_FLAG_USE_SM;
        pd.radiance += evalDirectAnalytic(sd, mi, pd.sg, pd.thp, sd.opacity, maskFlag, minTriangleSide) * sd.opacity;

        // Ambient light
        pd.radiance += pd.thp * bsdfProperties.diffuseReflectionAlbedo * kAmbientStrength * sd.opacity;

        //Update throughput
        float transparency = (1.0 - sd.opacity);
        float3 thp = pd.thp * transparency;
        if (all(thp <= kMinOpacity))
            pd.terminated = true;

        if(!rayData.isOpaque)
            pd.thp = thp;

        //Update ray
        ray.TMin = rayData.dist + 1e-7;
        rayData = TransparencyRayData(ray.TMax);
    }

    //Write data
    gOutputColor[pixel] += float4(pd.radiance, 1.0);
}

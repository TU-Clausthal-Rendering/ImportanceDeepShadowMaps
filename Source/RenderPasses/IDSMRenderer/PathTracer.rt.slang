#include "EvaluateAnalyticLight.slang"
 
import Utils.Math.MathHelpers;

cbuffer CB
{
    uint gFrameCount; // Current it for sample gen
}

// Outputs
RWTexture2D<float4> gOutputColor;

// Depths/MV for TAA or DLSS
RWTexture2D<float> gOutDepth;
RWTexture2D<float2> gOutMotion;

#define is_valid(name) (is_valid_##name != 0)
static const float kMinOpacity = MIN_OPACITY;
static const uint kMaxElements = MAX_ELEMENTS; // For now 256 max transparent elements on a ray
static const bool kAdjustShadingNormals = true;
static const bool kUseIrregularShadowMask = USE_IRRGEGULAR_SHADOW_MASK;

static const uint kMaxBounces = MAX_BOUNCES;           //Max Bounces
static const bool kUseRussianRoulette = USE_RUSSIAN_ROULETTE; //Russian Roulette

/** Payload for alpha ray (20B + 16B if lod is enabled)
*/
struct TransparencyRayData {
    PackedHitInfo hit; //< Packed Hit
    float dist;
    bool isOpaque;

    __init(float tMax)
    {
        hit = {}; // Init emtpy
        dist = tMax;
        isOpaque = false;
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
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

    return sd;
}

/** Russian Roulette function that can abort a path early
    Return "true" if the path should be aborted (absorbed)
*/
bool russianRoulette(inout float3 thp, inout SampleGenerator sg) {
    float prob = luminance(thp);
    float rnd = sampleNext1D(sg);
    bool rr = rnd > prob;
    if (!rr) {
        thp /= prob;
    }
    return rr;
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
        // Update payload
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

    // Prepare path data
    PathData pd = PathData();
    Ray cameraRayInfo = gScene.camera.computeRayPinhole(pixel, frameDim);
    pd.origin = cameraRayInfo.origin;
    pd.direction = cameraRayInfo.dir;

    // Set sampleGenerator
    pd.sg = SampleGenerator(pixel, gFrameCount);

    float tMax = gScene.camera.data.farZ;

    // Motion vector and depth
    PackedHitInfo packedMvecHit = {};
    float depth = 1;

    //Prepare Ray
    RayDesc ray;
    ray.Origin = pd.origin;
    ray.Direction = pd.direction;
    ray.TMin =  gScene.camera.data.nearZ;
    ray.TMax = tMax;
    TransparencyRayData rayData = TransparencyRayData(tMax);
    bool isPrimaryCameraRay = true;

    for(uint i=0; i<=kMaxBounces && !pd.terminated; i++){
        // Trace Ray
        uint rayFlags = 0;
        uint instanceMask = 0;
        if (isPrimaryCameraRay) {
            instanceMask = kInstanceCameraParticle;
        } else {
            instanceMask = getBestParticleShadowPlane(ray.Direction, true);
        }
        TraceRay(gScene.rtAccel, rayFlags, instanceMask, 0, rayTypeCount, 0, ray, rayData);

        //Miss
        if (rayData.dist < 0) {
            pd.radiance += gScene.envMap.eval(pd.direction) * pd.thp;
            pd.terminated = true;
            break;
        }

        HitInfo hit = HitInfo(rayData.hit);
        if(!hit.isValid())
            break;

        ITextureSampler lod = ExplicitLodTextureSampler(0.f);
        //Get shading data
        ShadingData sd = loadShadingData(hit, pd.direction, lod);
        // Get material instance
        uint hints = 0;
        if (kAdjustShadingNormals)
            hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        // Emissive lights
        BSDFProperties bsdfProperties = mi.getProperties(sd);
        pd.radiance += pd.thp * bsdfProperties.emission * sd.opacity;

        // Analytic
        uint maskFlag = TRANSPARENCY_MASK_FLAG_NONE;
        if (kUseIrregularShadowMask)
            maskFlag = rayData.isOpaque ? TRANSPARENCY_MASK_FLAG_USE_MASK : TRANSPARENCY_MASK_FLAG_USE_SM;
        pd.radiance += evalDirectAnalytic(sd, mi, pd.sg, pd.thp, sd.opacity, maskFlag) * sd.opacity;

        //Determine THP and new ray
        //if(i==0)
        //    ray.TMax = 100000.0;

        // Update depth and motion vector if valid
        if(isPrimaryCameraRay){
            if (is_valid(gOutDepth)) {
                float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(sd.posW, 1.f));
                depth = curPosH.z / curPosH.w;
            }
            if (is_valid(gOutMotion)) {
                packedMvecHit = hit.getData();
            }
        }
        
        //Calc new ray for next iteration
        float pReflect = sampleNext1D(pd.sg);
        if(pReflect < sd.opacity)
        {
            // Sample material.
            BSDFSample bsdfSample;
            pd.terminated |= !mi.sample(sd, pd.sg, bsdfSample);
            pd.origin = sd.computeNewRayOrigin(!bsdfSample.isLobe(LobeType::Transmission));
            pd.direction = bsdfSample.wo;
            pd.thp *= bsdfSample.weight;
            /* Cancels out
            pd.thp /= sd.opacity;
            pd.thp *= sd.opacity;
            */
            ray.TMin = 0.0;
            rayData = TransparencyRayData(ray.TMax);
            isPrimaryCameraRay = false;
        }
        else {
            /* Cancels out
            pd.thp /= (1.0 - sd.opacity);
            pd.thp *= (1.0 - sd.opacity);
            */
            ray.TMin = rayData.dist + 1e-10;
        }

        if (kUseRussianRoulette) {
            pd.terminated |= russianRoulette(pd.thp, pd.sg);
        }
        else {
            //Abort if 
            if (all(pd.thp <= kMinOpacity))
                pd.terminated = true;
        }

        //Update ray
        rayData = TransparencyRayData(ray.TMax);
        ray.Origin = pd.origin;
        ray.Direction = pd.direction;

    }

    //Calc Mvec
    float2 mVec = float2(0);
    if (is_valid(gOutMotion)) {
        HitInfo mVecHit = HitInfo(packedMvecHit);
        if(mVecHit.isValid()){
            // Compute motion vector.
            float3 prevPosW = gScene.getPrevPosW(mVecHit.getTriangleHit());
            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
            mVec = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
        }
    }

    //Write data
    gOutputColor[pixel] = float4(pd.radiance, 1.0);

    if (is_valid(gOutMotion))
        gOutMotion[pixel] = mVec;
    if (is_valid(gOutDepth))
        gOutDepth[pixel] = depth;
}

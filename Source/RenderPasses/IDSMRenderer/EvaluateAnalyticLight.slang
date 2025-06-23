#pragma once
#include "SharedShadowDefines.slang"
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Sampling.SampleGenerator;

import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Color.ColorHelpers;

import DSMAccelerationStructure.DSMAccelerationStructure;
import DSMLinkedList.DSMLinkedList;
import IDSMAccelerationStructure.IDSMAccelerationStructure;
import IDSMLinkedList.IDSMLinkedList;

// Shadow Eval Modes (see IDSMRenderer.h)
#define SHADOW_EVAL_MODE_RAYTRACING 0
#define SHADOW_EVAL_MODE_DSM_AS 1
#define SHADOW_EVAL_MODE_DSM_LL 2
#define SHADOW_EVAL_MODE_IDSM_AS 3
#define SHADOW_EVAL_MODE_IDSM_LL 4

#ifndef SHADOW_EVAL_MODE
#define SHADOW_EVAL_MODE SHADOW_EVAL_MODE_RAYTRACING
#endif

// LightSampleModes (see IDSMRenderer.h)
#define LIGHT_SAMPLE_MODE_UNIFORM 0
#define LIGHT_SAMPLE_MODE_RIS 1
#define LIGHT_SAMPLE_MODE_ALL 2

//Imporance modes (see IDSMRenderer.h)
#define IMPORTANCE_MODE_OPACITY 0
#define IMPORTANCE_MODE_OPACITY_THP 1
#define IMPORTANCE_MODE_THP 2
#define IMPORTANCE_MODE_THP_SQ 3
#define IMPORTANCE_MODE_BRDF_THP 4
#define IMPORTANCE_MODE_BRDF_THP_OPACITY 5
#define IMPORTANCE_MODE_UNIFORM 6

#ifndef IMPORTANCE_MODE
#define IMPORTANCE_MODE IMPORTANCE_MODE_OPACITY_THP
#endif

Buffer<uint> gParticleMaterials; //Buffer with the bitsize of all materials. Particle materials bits are set

static const float kMinIncidentAngle = 1e-3; // Min Incident angle for light
static const bool kUseStochasticRayTracing = USE_STOCHASTIC_RAY_TRACING; //Uses the stochastic version of the shadow ray
static const bool kTRUseColoredTransparency = TR_USE_COLORED_TRANSPARENCY;
static const bool kUseSoftShadows = USE_SOFT_SHADOWS;
static const float kSoftShadowsPosRadius = SOFT_SHADOWS_POS_RADIUS; // For Point and Spot
static const float kSoftShadowsDirSpread = SOFT_SHADOWS_DIR_SPREAD; // For Directional light
static const bool kIrregularShadowMaskUseRay = USE_IRRGEGULAR_SHADOW_MASK;
static const bool kShadowMaterialFlagAsBlacklist = SHADOW_MATERIAL_FLAG_AS_BLACKLIST; // Enables the shadow material flag as blacklist when mask is enabled
static const bool kUseDeepShadowRayOutsideOfSM = USE_DEEP_SHADOW_RAY_OUTSIDE_OF_SM; //Enables a deep shadow ray outside of the SM
static const bool kUseParticleRayOffset = false;


// For RIS light sampling
struct RISReservoir {
    uint lightIdx; // Light Index
    float wSum;    // Weight and Weight sum
    // uint M;         //Unused, as all possible Samples are seen

    /** Initializes ray payload with default parameters.
     */
    __init() {
        lightIdx = 0;
        wSum = 0.0;
        // M = 0;
    }

    [mutating] bool update(uint index, float risWeight, float rnd) {
        wSum += risWeight;
        // M += 1;
        bool selectSample = (rnd * wSum < risWeight);
        if (selectSample) {
            lightIdx = index;
        }
        return selectSample;
    }

    [mutating] void finalize(float targetFunction) {
        /* //When not looping through all lights
        if(targetFunction > 0 && M > 0){
            wSum = (1.0 / targetFunction) * ((1.0/M) * wSum);
            r.M = 1;
        */
        if (targetFunction > 0) {
            wSum = wSum / targetFunction;
        } else {
            wSum = 0;
        }
    }
};

/** Checks if the material is a particle
 */
bool isHitObjectAParticle(uint materialID) {
    uint bufferIdx = materialID / 32;
    uint bitPos = materialID % 32;
    uint bitIdx = 1u << bitPos;
    return (gParticleMaterials[bufferIdx] & bitIdx) > 0;
}

/** Samples a light index either uniform or with RIS (Good if a couple lights are used that dont overlap each other)
 */
AnalyticLightSample sampleLightIndex(float3 posW, uint lightCount, inout SampleGenerator sg, inout uint lightIndex, out float invPdf, out uint type) {
    AnalyticLightSample ls = {};
    #if LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_UNIFORM //1SPP uniform
        float lightSelect = 0.0;
        lightSelect = sampleNext1D(sg);

        // Pick one of the analytic light sources randomly with equal probability.
        lightIndex = min(uint(lightSelect * lightCount), lightCount - 1);
        invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

        // Sample local light source.
        LightData lightData = gScene.getLight(lightIndex);
        type = lightData.type;
        sampleLight(posW, lightData, sg, ls);
    #elif LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_RIS //RIS
        RISReservoir r = RISReservoir();
        // Loop through all lights
        for (uint i = 0; i < lightCount; i++) {
            float randomNum = i == 0 ? 0.5 : sampleNext1D(sg); // Random number
            LightData lightData = gScene.getLight(i);
            AnalyticLightSample itLs;
            sampleLight(posW, lightData, sg, itLs);
            float w = luminance(itLs.Li); // Take the Light attenuation as weight
            if (r.update(i, w, randomNum)) {
                ls = itLs;
                type = lightData.type;
            }
        }
        r.finalize(luminance(ls.Li)); // targetFunction
        // Out data
        lightIndex = r.lightIdx;
        invPdf = r.wSum; // W
    #else // LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_ALL //All in loop
        invPdf = 1.0;
        // Sample local light source.
        LightData lightData = gScene.getLight(lightIndex);
        type = lightData.type;
        sampleLight(posW, lightData, sg, ls);
    #endif

    return ls;
}

/** Traces a shadow ray towards a light source with ray queries
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from shading point towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \return Visibility in [0,1]
*/
float3 traceShadowRay(float3 origin, float3 dir, float3 lightDir, float distance, float tMin = 0.0, bool deepShadowRay = false)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0;
    ray.TMax = distance;

    float3 visibility = 1.0;

    RayQuery<RAY_FLAG_NONE> rayQuery;
    const uint instanceMask = kShadowMaterialFlagAsBlacklist && !deepShadowRay ? kInstanceNoParticle : getBestParticleShadowPlane(lightDir,true);
    rayQuery.TraceRayInline(gScene.rtAccel, 0, instanceMask, ray);

    while (rayQuery.Proceed()) {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            // Get shading data
            TriangleHit triangleHit;
            triangleHit.instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
            triangleHit.primitiveIndex = rayQuery.CandidatePrimitiveIndex();
            triangleHit.barycentrics = rayQuery.CandidateTriangleBarycentrics();

            VertexData v = gScene.getVertexData(triangleHit);
            uint materialID = gScene.getMaterialID(triangleHit.instanceID);
            ITextureSampler lod = ExplicitLodTextureSampler(0.f);

            if(kTRUseColoredTransparency){
                ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -ray.Direction, lod);
                //uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
                let mi = gScene.materials.getMaterialInstance(sd, lod);
                // Emissive lights
                BSDFProperties bsdfProperties = mi.getProperties(sd);
                float3 color = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo;
                //Normalize color
                float maxColor = max(max(color.x, color.y), color.z);
                color /= maxColor;
                if (sd.opacity < kMinOpacity || ((rayQuery.CandidateTriangleRayT() < tMin) && (sd.opacity < 1.0)))
                    visibility *= 1.0;
                else
                    visibility *= color * (1.0 - sd.opacity);
            }else{
                float opacity = gScene.materials.getMaterialOpacity(v, materialID, lod);
                if (rayQuery.CandidateTriangleRayT() >= tMin || opacity >= 1.0)
                    visibility *= saturate(1.0 - opacity);
            }
            
            if (all(visibility <= kMinOpacity))
                rayQuery.Abort();
        }
    }

    //In case of non opaque hit
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        visibility = 0.0;

    return visibility;
}

/** Traces a shadow ray towards a light source with ray queries. Uses the sample generator to randomly abort the ray
    \param[in] origin Ray origin for the shadow ray.
    \param[in] dir Direction from shading point towards the light source (normalized).
    \param[in] distance Distance to the light source.
    \param[inout] Sample Generator
    \return Visbility that is either 0 or 1
*/
float3 traceShadowRayStochastic(float3 origin, float3 dir,float3 lightDir, float distance, inout SampleGenerator sg, float TMin = 0.0, bool deepShadowRay = false)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = TMin;
    ray.TMax = distance;

    float3 visibility = 1.0; //Visible

    RayQuery<RAY_FLAG_NONE> rayQuery;
    const uint instanceMask = kShadowMaterialFlagAsBlacklist && !deepShadowRay ? kInstanceNoParticle : getBestParticleShadowPlane(lightDir, true);
    rayQuery.TraceRayInline(gScene.rtAccel, 0, instanceMask, ray);

    while (rayQuery.Proceed()) {
        if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            // Get shading data
            TriangleHit triangleHit;
            triangleHit.instanceID = GeometryInstanceID(rayQuery.CandidateInstanceID(), rayQuery.CandidateGeometryIndex());
            triangleHit.primitiveIndex = rayQuery.CandidatePrimitiveIndex();
            triangleHit.barycentrics = rayQuery.CandidateTriangleBarycentrics();

            VertexData v = gScene.getVertexData(triangleHit);
            uint materialID = gScene.getMaterialID(triangleHit.instanceID);
            // LOD mode handling TODO this is incorrect as it assumes the pixel is the camera
            ITextureSampler lod = ExplicitLodTextureSampler(0.f);

            if(kTRUseColoredTransparency){
                ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -ray.Direction, lod);
                //uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
                let mi = gScene.materials.getMaterialInstance(sd, lod);
                // Emissive lights
                BSDFProperties bsdfProperties = mi.getProperties(sd);
                float3 color = bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo;
                //Normalize color
                float maxColor = max(max(color.x, color.y), color.z);
                color /= maxColor;
                float3 sampleVis = 1.0;
                if(sd.opacity >= kMinOpacity)
                    sampleVis = color * (1.0 - sd.opacity);

                //Play russian roulette with every color channel
                float3 roulett = sampleNext3D(sg);
                for(uint i =0; i<3;i++){
                    if(roulett[i] > sampleVis[i])
                        visibility[i] = 0;
                }
                if(all(visibility <= 0))
                    rayQuery.Abort();
            }
            else //Singe channel version
            {
                float opacity = gScene.materials.getMaterialOpacity(v, materialID, lod);
                float roulett = sampleNext1D(sg);
                //Use russian roulette to set the color to zero
                if (roulett <= opacity) {
                    visibility = 0.0;
                    rayQuery.Abort();
                }
            }
           
        }
    }

    // No need to check comitted status, as no hit is accepted
    if (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        visibility = 0.0;

    return visibility;
}

/** Evaluates the direct illumination from analytic lights.
    This function samples Falcor's light list uniformly with one shadow ray.
    \param[in] sd Shading data.
    \param[in] mi Material instance.
    \param[in,out] sg SampleGenerator object.
    \return Outgoing radiance in view direction.
*/
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg, const float3 thp = float3(1.0), const float opacity = 1.0, uint maskFlags = TRANSPARENCY_MASK_FLAG_NONE, float minTriSide = 0.0)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return float3(0.0);

    // Pick one of the analytic light sources randomly with equal probability.
    uint lightIndex = 0;
    float invPdf = 1.0;
    uint lightType = 0;
    float3 Li = float3(0);

    #if LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_ALL // For all loop over all lights, for other methods 1SPP is evaluated
    for (lightIndex; lightIndex < lightCount; lightIndex++)
    #endif
    {
        // Sample local light source.
        AnalyticLightSample ls = sampleLightIndex(sd.posW, lightCount, sg, lightIndex, invPdf, lightType);
        float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
        
        // Flip light dir for non-opaque surfaces, so light shines through
        bool flipLightDir = (NdotL < 0) && (sd.opacity < 1);
        float3 lightDirection = flipLightDir ? -ls.dir : ls.dir;
        if (flipLightDir)
            ls.Li *= (1.0 - sd.opacity); // consider the opacity of the hit if light shined through

        bool isParticle = isHitObjectAParticle(sd.materialID);
        if (isParticle)
            lightDirection = sd.frontFacing ? sd.faceN : -sd.faceN;
           
        if (!isParticle || !kUseParticleRayOffset)
            minTriSide = 0.0;

        // Reject sample if invalid or has an too steep incident angle
        if (all(ls.Li <= 0.0) || (abs(NdotL) < kMinIncidentAngle) && !isParticle) {
            #if LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_ALL // Loop over all lights
                    continue;
            #else
                    return float3(0.0);
            #endif
        }

        // Eval BSDF
        float3 brdf = mi.eval(sd, lightDirection, sg) * thp;
        if (all(brdf <= 0)) {
            #if LIGHT_SAMPLE_MODE == LIGHT_SAMPLE_MODE_ALL // Loop over all lights
                continue;
            #else
                return float3(0.0);
            #endif
        }

        // Importance between 1 and 0
        float importance = 0.0;
        #if IMPORTANCE_MODE == IMPORTANCE_MODE_OPACITY
            importance = opacity;
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_OPACITY_THP
            importance = luminance(thp) * opacity;
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_THP
            importance = luminance(thp);
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_THP_SQ
            importance = luminance(thp * thp);
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_BRDF_THP
            importance = luminance(thp * brdf);
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_BRDF_THP_OPACITY
            importance = luminance(thp * brdf) * opacity;
        #elif IMPORTANCE_MODE == IMPORTANCE_MODE_UNIFORM
            importance = 0; //Gets capped to lowest value
        #endif
        importance = saturate(importance);

        // Test visibility by tracing a shadow ray.
        bool isDirectional = lightType == (uint)LightType::Directional; // Slight different calculations for ortho
        float3 visibility = float3(1.0);

        // Shadow maps
        #if SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_DSM_AS
            visibility = gDSMAccelerationStructure.eval(lightIndex, sd.posW, isDirectional,maskFlags);
        #elif SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_DSM_LL
            visibility = gDSMLinkedList.eval(lightIndex, sd.posW, isDirectional, maskFlags);
        #elif SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_IDSM_AS
            visibility = gIDSMAccelerationStructure.eval(lightIndex, sd.posW, isDirectional, importance, maskFlags);
        #elif SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_IDSM_LL
            visibility = gIDSMLinkedList.eval(lightIndex, sd.posW, isDirectional, importance, maskFlags);
        #endif

        bool useMaskRay = kIrregularShadowMaskUseRay && (maskFlags & TRANSPARENCY_MASK_FLAG_USE_MASK) != 0;
        bool sampleOutsideSM = (visibility.x < 0);
        // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
        if (SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_RAYTRACING || useMaskRay || sampleOutsideSM) {
            if (visibility.x < 0)
                visibility = 1.0;
            const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);
            if (kUseSoftShadows) {
                if (isDirectional) {
                    ls.dir = random_spread_direction(ls.dir, kSoftShadowsDirSpread, sampleNext2D(sg));
                } else {
                    ls.posW += sample_sphere(sampleNext2D(sg)) * kSoftShadowsPosRadius;
                    ls.dir = ls.posW - origin;
                    ls.distance = length(ls.dir);
                    ls.dir /= ls.distance;
                }
            }
            float3 rayVisibility = 1.0;
            bool shootDeepRay = SHADOW_EVAL_MODE == SHADOW_EVAL_MODE_RAYTRACING || (sampleOutsideSM && kUseDeepShadowRayOutsideOfSM);
            if (kUseStochasticRayTracing)
                rayVisibility = traceShadowRayStochastic(origin, ls.dir, ls.normalW, ls.distance, sg, minTriSide / 2.0, shootDeepRay);
            else
                rayVisibility = traceShadowRay(origin, ls.dir, ls.normalW, ls.distance, minTriSide / 2.0, shootDeepRay);
            visibility = min(visibility, rayVisibility);
        }

        // Clamp shadow in [0,1]
        visibility = clamp(visibility, 0.0, 1.0);

        // Evaluate contribution.
        Li += brdf * visibility * ls.Li * invPdf;
    }

    return Li;
}

/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

#include "Rendering/ShadowMaps/ShadowMap.h"
#include "DeepShadowMapMethod.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "IDSMMaskAndOpaqueShadowMap/IDSMMaskAndOpaqueShadowMap.h"

using namespace Falcor;

class IDSMRenderer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(IDSMRenderer, "IDSMRenderer", "Renderer for scenes with Transparencies");

    static ref<IDSMRenderer> create(ref<Device> pDevice, const Properties& props) { return make_ref<IDSMRenderer>(pDevice, props); }

    IDSMRenderer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    //Possible shadow render methods
    enum class ShadowRenderMethod : uint
    {
        RayTracing = 0,         //Deep shadow ray
        DSM_AS = 1,             //Deep Shadow Map with Acceleration Structure
        DSM_LL = 2,             //Deep Shadow Map with Linked List
        IDSM_AS = 3,            //Importance Deep Shadow Map with Acceleration Structure
        IDSM_LL = 4             //Importance Deep Shadow Map with Linked List
    };

    FALCOR_ENUM_INFO(ShadowRenderMethod,  {
            {ShadowRenderMethod::RayTracing, "RayTracing"},
            {ShadowRenderMethod::IDSM_AS, "IDSM-AS"},
            {ShadowRenderMethod::IDSM_LL, "IDSM-LL"},
            {ShadowRenderMethod::DSM_AS, "DSM-AS"},
            {ShadowRenderMethod::DSM_LL, "DSM-LL"},
        }
    );

    //Light sample mode for analytic lights (see EvaluateAnalyticLight.slang)
    enum class LightSampleMode : uint
    {
        Uniform = 0,
        RIS = 1,
        All = 2,
    };

    FALCOR_ENUM_INFO(
        LightSampleMode,
        {
            {LightSampleMode::Uniform, "Uniform"},
            {LightSampleMode::RIS, "RIS"},
            {LightSampleMode::All, "All"},
        }
    );

    //Importance mode for IDSM (see EvaluateAnalyticLight.slang)
    enum class ImportanceMode : uint
    {
        Opacity = 0,
        Opacity_Thp = 1,
        Thp = 2,
        Thp_Thp = 3,
        Brdf_Thp = 4,
        Brdf_Opacity_Thp = 5,
        Uniform = 6,
    };

    FALCOR_ENUM_INFO(
        ImportanceMode,
        {
            {ImportanceMode::Opacity, "HitOpacity"},
            {ImportanceMode::Opacity_Thp, "HitOpacity*Visibility"},
            {ImportanceMode::Thp, "Visibility"},
            {ImportanceMode::Thp_Thp, "Visibility*Visibility"},
            {ImportanceMode::Brdf_Thp, "BRDF*Visibility"},
            {ImportanceMode::Brdf_Opacity_Thp, "BRDF*HitOpacity*Visibility"},
            {ImportanceMode::Uniform, "Unweighted(Uniform)"},
        }
    );

    //Renderers
    enum class CameraRenderMode : uint
    {
        DirectRT= 0,
        DirectRT_Reflections = 1,
        PathTracer = 2
    };

    FALCOR_ENUM_INFO(
        CameraRenderMode,
        {
            {CameraRenderMode::DirectRT, "DirectRT"},
            {CameraRenderMode::DirectRT_Reflections, "DirectRT+RayReflections"},
            {CameraRenderMode::PathTracer, "PathTracer"},
        }
    );

    enum class CamJitterSamplePattern : uint32_t
    {
        Center,
        DirectX,
        Halton,
        Stratified,
    };

    FALCOR_ENUM_INFO(
        CamJitterSamplePattern,
        {
            {CamJitterSamplePattern::Center, "Center"},
            {CamJitterSamplePattern::DirectX, "DirectX"},
            {CamJitterSamplePattern::Halton, "Halton"},
            {CamJitterSamplePattern::Stratified, "Stratified"},
        }
    );

private:
    // Properties for render graph
    void parseProperties(const Properties& props); 
    //Defines for the light evaluation. Can update every frame
    DefineList getLightEvalDefines();

    //Prepare additional textures and buffers
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    //Evaluates the transparencies until the first opaque surface
    void evalDirectTransparency(RenderContext* pRenderContext, const RenderData& renderData);
    //Ray Traced Reflections
    void evalRayReflections(RenderContext* pRenderContext, const RenderData& renderData);
    //Path tracing pass
    void evalPathTracer(RenderContext* pRenderContext, const RenderData& renderData);

    //Camera Jitter
    void updateFrameDim(const uint2 frameDim);
    void updateSamplePattern();

    // Internal state
    ref<Scene> mpScene;                                     ///< Current scene.
    ref<SampleGenerator> mpSampleGenerator;                 ///< GPU sample generator.
    ref<CPUSampleGenerator> mpCameraJitterGenerator;        ///< Sample generator for camera jitter.
    std::shared_ptr<IDSMMaskAndOpaqueShadowMap> mpIDSMMask;    ///< Shadow Mask to distribute rays only on non-opaque objects for DSM techniques

    CameraRenderMode mCameraRenderMode = CameraRenderMode::DirectRT;
    ShadowRenderMethod mShadowRenderMethod = ShadowRenderMethod::IDSM_AS;
    uint mSelectedShadowMethod = std::max((int)mShadowRenderMethod - 1, 0);

    std::vector<std::shared_ptr<DeepShadowMapMethod>> mShadowMethods; // Shadow Methods that rely on extra structures (mSelectedShadowMethod - 1)

    // Runtime data Tracer
    uint mFrameCount = 0; ///< Frame count since scene was loaded.
    uint2 mRenderDims = uint2(512);
    LightSampleMode mLightSampleMode = LightSampleMode::All;
    bool mOptionsChanged = false;
    bool mShadowUseStochasticRayTracing = false; //Enable stochastic ray tracing for visibility
    bool mUseDeepShadowRayOutsideOfShadowMap = false;   //Shoots a deep shadow ray if the sample area is not covered by the deep shadow map
    ImportanceMode mImportanceMode = ImportanceMode::Opacity_Thp;

    //Camera Jitter
    CamJitterSamplePattern mCameraJitterSamplePattern = CamJitterSamplePattern::Halton;
    uint mCameraJitterNumSamples = 16;

    //Reflections
    float mRayReflectionsRoughnessThreshold = 0.7f; //Threshold for ray reflections

    //Shadow Mask
    bool mUseNonOpaqueShadowMask = true;            //Enables a shadow mask for non-opaque objects
    uint mMaskISMMultFactor = 9u;                  // Mult factor for ISM 
    bool mUseShadowMaterialFlagAsBlacklist = true;     //Uses the non-shadow throwable flag as blacklist for non-opaque objects

    DeepShadowMapMethod::GlobalShadowSettings mShadowSettings = {};

    //Path Tracer specific settings
    uint mPTMaxBounces = 1024;
    bool mPTUseRussianRoulette = true;

    //Shading Settings
    float mAmbientStrength = 0.25f;
    float mEnvMapStrength = 1.f;

    //Buffer/Textures
    ref<Texture> mpTransparencyThp; //Thp texture for transparency
    ref<Texture> mpReflectionsMask; //Mask where ray reflections should be used
    ref<Texture> mpReflectionsHit; //Opaque hit for the reflection. Is needed when V-Buffer is not used
    ref<Buffer> mpParticleMaterials; //Buffer that stores if the material with index x is a particle

    //Passes
    // Pipelines / Programs
    struct RayTracingPipeline
    {
        ref<RtProgram> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        void resetPip()
        {
            pProgram.reset();
            pBindingTable.reset();
            pVars.reset();
        }
    };

    RayTracingPipeline mEvalTransparencyDirectRay; // Ray Tracing pass for evaluating the Transparencies along the primary ray
    ref<ComputePass> mpEvalDirectPass; //Compute Pass for direct light
    RayTracingPipeline mTransparencyPathTracer; //Pipeline for the path tracer
    RayTracingPipeline mReflectionsPass;
};

FALCOR_ENUM_REGISTER(IDSMRenderer::ShadowRenderMethod);
FALCOR_ENUM_REGISTER(IDSMRenderer::LightSampleMode);
FALCOR_ENUM_REGISTER(IDSMRenderer::CameraRenderMode);
FALCOR_ENUM_REGISTER(IDSMRenderer::ImportanceMode);
FALCOR_ENUM_REGISTER(IDSMRenderer::CamJitterSamplePattern);

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
#include "Rendering/Materials/TexLODTypes.slang"

using namespace Falcor;

class DeepShadowMapMethod
{
public:
    virtual ~DeepShadowMapMethod() = default;

    // Light MVP
    struct LightMVP
    {
        float3 pos = float3(0);
        float spreadAngle = 0;
        float4x4 view = float4x4();
        float4x4 projection = float4x4();
        float4x4 viewProjection = float4x4();
        float4x4 invViewProjection = float4x4();
        float4x4 invProjection = float4x4();
        float4x4 invView = float4x4();
    };

    enum class SMSamplePattern : uint
    {
        Center = 0,
        PerSampleHalton = 1,
        MatrixDirectX = 2,
        MatrixHalton = 3,
        MatrixStratified = 4,
    };

    FALCOR_ENUM_INFO(
        SMSamplePattern,
        {
            {SMSamplePattern::Center, "Center"},
            {SMSamplePattern::PerSampleHalton, "PerSampleHalton"},
            {SMSamplePattern::MatrixDirectX, "MatrixDirectX"},
            {SMSamplePattern::MatrixHalton, "MatrixHalton"},
            {SMSamplePattern::MatrixStratified, "MatrixStratified"},
        }
    );

    //Global shadow settings valid for every method
    struct GlobalShadowSettings
    {
        uint resolution = 512u;
        float2 nearFar = float2(0.1f, 60.f);
        float dirLightRange = 20.f;
        float midpointPercentage = 0.5f;
        float depthBias = 1e-3f;
        bool enableColoredTransparency = false;
        bool enableSoftShadows = false;
        float softShadowsPositionRadius = 0.001f;
        float softShadowsDirectionsSpread = 1.f;
        bool dirLightPutCameraOnGrid = true;
        SMSamplePattern samplePattern = SMSamplePattern::MatrixHalton;
        uint jitterSampleCount = 16;
    };

    /** Generate resources needed to evaluate the Shadow Method (e.g. Shadow Map)
    * Should be called every frame and needs to be called before using any resources from that pass
    */
    virtual void generate(RenderContext* pRenderContext, const RenderData& renderData) {}

    /** Returns defines needed to evaluate the method
     */
    virtual DefineList getDefines();

    /** Set the needed shader data for the method (textures,buffer, etc)
    */
    virtual void setShaderData(const ShaderVar& var) {}

    /** Some methods can use an additional mask, this is set here
    */
    virtual void setShadowMask(const ShaderVar& var, ref<Texture> maskTex, ref<Texture> maskSM, bool enable = true) {}
    virtual void setShadowMask(const ShaderVar& var, ref<Texture> maskTex, std::vector<ref<Buffer>> &maskISM, bool enable = true) {}

    /** Render UI for the method
    */
    virtual bool renderUI(Gui::Widgets& widget) { return false; }

    /** Optional Debug pass.
        It should be called every frame, so if debug is disabled, the function should return before doing any computationally expensive work.
    */
    virtual void debugPass(RenderContext* pRenderContext, const RenderData& renderData, ref<Texture> debugOut = nullptr, ref<Texture> colorOut = nullptr) {}

    /* Set Colored Transparency mode
    */
    void setColoredTransparency(const bool enabled) { mUseColoredTransparency = enabled; }

    /* Set Near Far for lights
     */
    void setNearFar(const float2 nearFar);

    /** Set Soft shadow parameters
    */
    void setSoftShadowParameter(bool enabled, float positionRadius, float directionalSpread)
    {
        mEnableRandomSoftShadows = enabled;
        mRandomSoftShadowsPositionRadius = positionRadius;
        mRandomSoftShadowsDirSpread = directionalSpread;
    }

    /** Set Soft shadow parameters
     */
    const void getSoftShadowParameter (bool& isEnabled, float& positionRadius, float& directionalSpread) const
    {
        isEnabled = mEnableRandomSoftShadows;
        positionRadius = mRandomSoftShadowsPositionRadius;
        directionalSpread = mRandomSoftShadowsDirSpread;
    }

    /** Gets current shadow map resolution
    */
    const uint2 getShadowMapResolution() const { return mResolution; }

    /** Sets the blacklist flag
    */
    void enableBlacklist(bool enable) { mEnableBlacklistWithShadowMaterialFlag = enable; }

    /** Get the light MVPS for the scene
    */
    const std::vector<LightMVP>& getLightMVPs() const { return mShadowMapMVP; }

    /* For irregular methods only. Returns the sample distribution
    */
    virtual const std::vector<ref<Texture>>* getSamplesDistribution() const { return nullptr; };

    /* Gets dispatch size for the gen shader
    */
    virtual const uint2 getShaderDispatchSize() const { return mResolution; }

     /* Get Sample distribution buffer. Can be nullptr
     */
    const ref<Buffer> getPerSampleJitterBuffer() const { return mpHaltonBuffer; }

    /* UI for the settings shared by every deep shadow map method
     */
    bool globalSettingsRenderUI(Gui::Widgets& widget, GlobalShadowSettings& settings);

    /* Setter for the settings struct
     */
    void setGlobalShadowSettings(GlobalShadowSettings& settings);
protected:
    //Init constants for the blur
    static const uint kBlurKernelWidthInit = 5;
    static const bool kBlurSigmaInit = 1.f;

    DeepShadowMapMethod(ref<Device> pDevice, ref<Scene> pScene);

    //Function to update the Shadow Map Matrices
    virtual void updateSMMatrices(bool rebuild = false);

    virtual void updateViewProjection(LightMVP& lightMVP, ref<Light> pLight);

    virtual void updateMVPAndJitter(LightMVP& lightMVP);

    void updateJitterSamplePattern();

    ref<Device> mpDevice;
    ref<Scene> mpScene;
    bool mHasDirectionalLight = false;      

    float mMidpointPercentage = 0.6f;     // Percentage where the midpoint is set. 0.5 is normal midpointSM, 0 is SM without bias
    float mMidpointDepthBias = 1e-2f; // Depth bias applied to tmin after a opaque hit. Scaled with pixel size. Normally 1e-7 is used
    uint2 mResolution = uint2(512);
    bool mUpdateSMMatrices = false;         //True if VP Matrices of the shadow maps should be recalculated
    bool mUpdateDirectional = true;         //To disable update of directional lights (for debug purposes)
    float2 mNearFar = float2(1.f, 60.f);    //Near and far for spot
    bool mResolutionChanged = false;         //True if the resolution changed
    bool mUseColoredTransparency = false;   //Enable colored transparency
    float2 mJitter = float2(0, 0);          //Optional Light Camera Jitter

    float mDirLightSMRange = 50.f;
    bool mDirLightSMPutOnCameraGrid = true;

    //Jitter
    SMSamplePattern mSamplePattern = SMSamplePattern::MatrixHalton; // Sample Pattern
    ref<CPUSampleGenerator> mpCPUSampleGenerator; ///< Sample generator for uniform camera jitter.
    uint mJitterSampleCount = 16;            // Number of jitter samples
    ref<Buffer> mpHaltonBuffer;             //  GPU per-sample halton buffer

    std::vector<LightMVP> mShadowMapMVP;    //Collection of all possible view/projection matrices from each light

    //Blacklist
    bool mEnableBlacklistWithShadowMaterialFlag = false;

    // Random Soft Shadows
    bool mEnableRandomSoftShadows = false;          // Enables Random offset of start position for shadow maps
    float mRandomSoftShadowsPositionRadius = 0.01f; // Random Radius for the start position
    float mRandomSoftShadowsDirSpread = 1.f;        // Pixel radius on far plane for spread

    //Pipelines / Programs
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

    struct RasterPipeline
    {
        ref<GraphicsState> pState;
        ref<GraphicsProgram> pProgram;
        ref<GraphicsVars> pVars;
        ref<Fbo> pFBO;

        void resetPip()
        {
            pState.reset();
            pProgram.reset();
            pVars.reset();
            pFBO.reset();
        }
    };
};

FALCOR_ENUM_REGISTER(DeepShadowMapMethod::SMSamplePattern);

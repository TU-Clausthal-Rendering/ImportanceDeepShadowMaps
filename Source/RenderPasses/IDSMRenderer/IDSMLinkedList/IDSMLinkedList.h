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
#include "../DeepShadowMapMethod.h"
#include "../ImportanceMapHelpers/ImportanceMapHelper.h"

class IDSMLinkedList : public DeepShadowMapMethod
{
public:
    virtual ~IDSMLinkedList() = default;

    IDSMLinkedList(ref<Device> pDevice, ref<Scene> pScene);

    /** Generate resources needed to evaluate the Shadow Method
     */
    virtual void generate(RenderContext* pRenderContext, const RenderData& renderData) override;

    /** Returns defines needed for the method
     */
    virtual DefineList getDefines() override;

    /** Set the needed shader data for the method (textures,buffer, etc)
     */
    virtual void setShaderData(const ShaderVar& var) override;

    /** Additional mask to reject the backprojectio
    */
    virtual void setShadowMask(const ShaderVar& var, ref<Texture> maskTex, std::vector<ref<Buffer>>& maskISM, bool enable = true) override;

    /** Render UI for the method
     */
    virtual bool renderUI(Gui::Widgets& widget) override;

    /** Optional Debug pass
    */
    virtual void debugPass(RenderContext* pRenderContext, const RenderData& renderData,  ref<Texture> debugOut = nullptr, ref<Texture> mask = nullptr) override;

    /*  Returns the sample distribution
    */
    virtual const std::vector<ref<Texture>>* getSamplesDistribution() const override { return mpImportanceMapHelper->getSamplesDistributionVector();};

    /* Gets dispatch size for the gen shader
     */
    virtual const uint2 getShaderDispatchSize() const override { return uint2(float2(mResolution) * mSampleOverestimate); }
private:
    void prepareResources(RenderContext* pRenderContext);

    // Function that generates the profiler passes in case they are not executed this frame
    void dummyProfileGeneration(RenderContext* pRenderContext);

    // Importance Map
    std::unique_ptr<ImportanceMapHelper> mpImportanceMapHelper;

    //Runtime
    uint mFrameCount = 0;
    bool mUseMask = false;

    //Sync Resources
    static const uint kFramesInFlight = 3; ///< Number of frames in flight for GPU/CPU sync
    ref<GpuFence> mpFence;                 ///< Fence for CPU/GPU syncs
    uint mStagingCount = 0;

    //Sample Gen
    float mSampleOverestimate = 1.00f; //How many more pixels are dispatched than the size of the shadow map. Only used with the optimized sample distribution

    //Shadow settings
    uint mApproxNumElementsPerPixel = 4u;
    std::vector<uint> mUIElementCounter;
    std::vector<uint64_t> mCounterFenceWaitValues; // Fence values forCounter sync
    uint mLinkedListNodeBufferSize = 0;
    uint mLinkedListDataFormatSize = 3; //TODO set automatically
    bool mRebuildDataBuffer = true;
    bool mAccelUsePCF = false;
    bool mTransparencyBufferUsesColor = false; // Checks if the transparency buffer data size matches the global setting

    uint mSkipFrameCount = 0;           // Counter for skipping frames
    uint mSkipGenerationFrameCount = 1; // Number of generated frames is 1/X

    LightMVP mStaggeredDirectionalLightMVP = {};
    int mDirectionalLightIndex = -1; //Used to set LightMVP

    //Debug
    bool mDebugEnableShowImportance = false;
    uint mDebugSelectedLight = 0;
    float mDebugScaleFactorIM = 3000.f;   //Impoartance map
    float mDebugScaleFactorSD = 120.f;   //sample distribution
    uint mDebugSelectedMipLevel = 0;
    
    ref<Sampler> mpLinearSampler;
    ref<Sampler> mpPointSampler;
    ref<SampleGenerator> mpSampleGenerator;

    std::vector<ref<Buffer>> mLinkedListCounter;                              // Counter for inserting points
    std::vector<ref<Buffer>> mLinkedListCounterCPU;                           // Counter for inserting points
    std::vector<ref<Buffer>> mLinkedListData;                                 // Transparency Data
    
    ref<ComputePass> mpDebugShowImportancePass;     //Debug show importance map or sample distribution
    RayTracingPipeline mGenLinkedListShadowPip; //RayTracingPipeline
};

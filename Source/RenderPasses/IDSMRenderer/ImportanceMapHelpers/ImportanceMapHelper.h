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
#include "Rendering/ShadowMaps/Blur/SMGaussianBlur.h"
#include "RenderGraph/RenderPass.h"


using namespace Falcor;

class ImportanceMapHelper
{
public:
    virtual ~ImportanceMapHelper() = default;

    ImportanceMapHelper(ref<Device> pDevice, uint lightCount, uint2 resolution, bool isLinkedList = false)
        : mpDevice(pDevice), mLightCount(lightCount), mResolution(resolution), mIsLinkedList(isLinkedList)
    {}

    void generateSampleDistribution(
        RenderContext* pRenderContext,
        ref<Buffer>& pLastFrameNodeCounter,
        uint maxNodeBufferSize,
        uint maxRayBudget,
        ref<Buffer>& pCurrentFrameNodeCounter,
        bool oneBufferForAllLights
    );
        
    void updateResolution(uint2 newResolution);

    void dummyRenderPassProfile(RenderContext* pRenderContext);

    void renderUI(Gui::Widgets& widget);

    ref<Texture>& getImportanceMap(uint index) { return mImportanceMap[index]; }
    ref<Texture>& getSampleDistribution(uint index) { return mSampleDistributions[index]; }

    const std::vector<ref<Texture>>* getSamplesDistributionVector() const { return &mSampleDistributions; }

private:
    void prepareResources();

    void reduceImportanceMapPass(RenderContext* pRenderContext);

    void distributeBudgetPass(
        RenderContext* pRenderContext,
        ref<Buffer>& pLastFrameNodeCounter,
        uint maxNodeBufferSize,
        uint maxRayBudget,
        bool oneBufferForAllLights
    );

    void genSampleDistributionPass(RenderContext* pRenderContext, ref<Buffer>& pCurrentFrameNodeCounter);

    //Pointer
    ref<Device> mpDevice;
    std::unique_ptr<SMGaussianBlur> mpGaussianBlur;

    //Variables
    bool mIsLinkedList = false;     //The counter needs to be cleared to the number of ray samples for the head buffer
    uint mLightCount = 1;
    uint2 mResolution = uint2(512);
    bool mResetResources = false;
    bool mUseReduceMipMap = false;  //If enabled reduce with mipmap chain is used
    bool mUseBlur = true;

    //Dynamic ray sample count change
    bool mEnableDynamicRayCountCalc = true;
    bool mResetRayCount = true;
    float mDynRCGuardPercentage = 0.85f;                // 15% buffer for possible changes
    float2 mDynRCChangePercentage = float2(0.1f, 0.6f); //(Increase/Decrease) Percentage. How much of the total difference should be used to increase/decrease number of samples


    //Resources
    std::vector<ref<Texture>> mImportanceMap;      // Importance Map
    std::vector<ref<Texture>> mSampleDistributions; // Sample Disribution
    ref<Buffer> mpLastFrameTotalRayBudget;          // Buffer to store the sample distribution from last frame. Used with Optimize Sample
                                           
    //Compute Passes
    ref<ComputePass> mpReducePass;                  // Reduce with Compute Shader
    ref<ComputePass> mpReduceMipMapPass;            // Reduce with Mipmap chain generation
    ref<ComputePass> mpDistributeBudgetPass;        // Distribute Budget with Importance Map
    ref<ComputePass> mpGenSampleDistributionPass;   // Generates the Sample Distribution from the budget
};


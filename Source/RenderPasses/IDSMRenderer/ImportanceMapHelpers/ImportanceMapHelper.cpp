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
#include "ImportanceMapHelper.h"
#include "Utils/Math/FalcorMath.h"

namespace
{
// Shader Paths
const std::string kShaderFolder = "RenderPasses/IDSMRenderer/ImportanceMapHelpers/";
const std::string kReduceMipMapShader = kShaderFolder + "ReduceMipMaps.cs.slang";
const std::string kReduceShader = kShaderFolder + "Reduce.cs.slang";
const std::string kDistributeBudgetShader = kShaderFolder + "DistributeBudget.cs.slang";
const std::string kGenSampleDistributionShader = kShaderFolder + "GenSampleDistribution.cs.slang";
}; // namespace

void ImportanceMapHelper::generateSampleDistribution(
    RenderContext* pRenderContext,
    ref<Buffer>& pLastFrameNodeCounter,
    uint maxNodeBufferSize,
    uint maxRayBudget,
    ref<Buffer>& pCurrentFrameNodeCounter,
    bool oneBufferForAllLights
)
{
    //Update Textures and Buffers
    prepareResources();

    //Get the Total Importance 
    reduceImportanceMapPass(pRenderContext);

    //Distribute Budget
    distributeBudgetPass(pRenderContext, pLastFrameNodeCounter, maxNodeBufferSize, maxRayBudget, oneBufferForAllLights);

    //Blur
    if (mUseBlur)
    {
        if(!mpGaussianBlur)
        {
            mpGaussianBlur = std::make_unique<SMGaussianBlur>(mpDevice);
            mpGaussianBlur->setBlurKernel(5, 1.f);
        }

        for (uint i = 0; i < mSampleDistributions.size(); i++)
            mpGaussianBlur->execute(pRenderContext, mSampleDistributions[i]); 
    }

    genSampleDistributionPass(pRenderContext, pCurrentFrameNodeCounter);
}

void ImportanceMapHelper::updateResolution(uint2 newResolution) {
    if (newResolution.x != mResolution.x || newResolution.y != mResolution.y)
    {
        mResolution = newResolution;
        mResetResources = true;
    }
}

void ImportanceMapHelper::prepareResources() {
    if (mResetResources)
    {
        mImportanceMap.clear();
        mSampleDistributions.clear();
        mpLastFrameTotalRayBudget.reset();
        mResetResources = false;
        mResetRayCount = true;
    }

    if (mImportanceMap.empty())
    {
        mImportanceMap.resize(mLightCount);
        for (uint i = 0; i < mLightCount; i++)
        {
            mImportanceMap[i] = Texture::create2D(
                mpDevice, mResolution.x, mResolution.y, ResourceFormat::R32Uint, 1u, Texture::kMaxPossible, nullptr,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
            mImportanceMap[i]->setName("ImportanceMap" + std::to_string(i));
        }
    }

    if (mSampleDistributions.empty())
    {
        mSampleDistributions.resize(mLightCount);
        for (uint i = 0; i < mLightCount; i++)
        {
            mSampleDistributions[i] = Texture::create2D(
                mpDevice, mResolution.x, mResolution.y, ResourceFormat::R32Float, 1u, Texture::kMaxPossible, nullptr,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
            mSampleDistributions[i]->setName("SampleDistribution" + std::to_string(i));
        }
    }

    if (!mpLastFrameTotalRayBudget)
    {
        std::vector<uint> initData(mLightCount, 0);
        mpLastFrameTotalRayBudget = Buffer::create(
            mpDevice, sizeof(uint) * mLightCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, initData.data()
        );
        mpLastFrameTotalRayBudget->setName("LastFrameTotalRayBudget");
    }
}

void ImportanceMapHelper::dummyRenderPassProfile(RenderContext* pRenderContext)
{
    if (mUseReduceMipMap)
        FALCOR_PROFILE(pRenderContext, "ImportanceMapReduce_MipMap");
    else
        FALCOR_PROFILE(pRenderContext, "ImportanceMapReduce");

    {
        FALCOR_PROFILE(pRenderContext, "DistributeSampleBudget");
    }

    if (mpGaussianBlur && mUseBlur)
    {
        mpGaussianBlur->profileDummy(pRenderContext); 
    }

    {
        FALCOR_PROFILE(pRenderContext, "GenSampleDistribution");
    }
}

void ImportanceMapHelper::renderUI(Gui::Widgets& widget) {

    //mResetRayCount |= widget.checkbox("Use GPU Sample Distribution optimization", mEnableDynamicRayCountCalc); //TODO this seems bugged
    if (mEnableDynamicRayCountCalc)
    {
        widget.var("GPU SD Fill Guard", mDynRCGuardPercentage, 0.001f, 1.f);
        widget.tooltip("Data/Node Buffer should be held around this fill percentage. Sample Distribution budget is dynamically changed to retain this fillrate.");
        widget.var("GPU SD Change Mult (Increase/Decrease)", mDynRCChangePercentage, 0.001f, 1.f);
        widget.tooltip(
            "Multiplier for the change value in the Sample Distribution. There is a different value for increase and decrease. "
            "Increase should be handled more conservatively, while the decrease should be quiet aggressive"
        );
    }

    widget.checkbox("Blur Sample Distribution", mUseBlur);
    if (mUseBlur && mpGaussianBlur)
    {
        if (auto group = widget.group("Blur Options"))
            mpGaussianBlur->renderUI(group);
    }

    widget.checkbox("Reduce Use Mipmap version", mUseReduceMipMap);
    widget.tooltip("Replaces Reduce with mipmap chain. Produces the same result as the reduce but is generally slower.");
}

void ImportanceMapHelper::reduceImportanceMapPass(RenderContext* pRenderContext) {
    if (mUseReduceMipMap)
    {
        FALCOR_PROFILE(pRenderContext, "ImportanceMapReduce_MipMap");
        // Create Gen Mips pass
        if (!mpReduceMipMapPass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kReduceMipMapShader).csEntry("main").setShaderModel("6_6");

            DefineList defines;
            defines.add("COUNT_LIGHTS", std::to_string(mLightCount));

            mpReduceMipMapPass = ComputePass::create(mpDevice, desc, defines, true);
        }

        for (uint m = 0; m < mImportanceMap[0]->getMipCount() - 1; m++)
        {
            auto var = mpReduceMipMapPass->getRootVar();
            for (uint i = 0; i < mLightCount; i++)
            {
                var["gSrc"][i].setSrv(mImportanceMap[i]->getSRV(m, 1u));
                var["gDst"][i].setUav(mImportanceMap[i]->getUAV(m + 1));
            }

            uint3 dispatchDim = uint3(mImportanceMap[0]->getWidth(m + 1), mImportanceMap[0]->getHeight(m + 1), mLightCount);
            var["CB"]["gDstSize"] = dispatchDim.xy();

            mpReduceMipMapPass->execute(pRenderContext, dispatchDim);
        }
    }
    else
    {
        FALCOR_PROFILE(pRenderContext, "ImportanceMapReduce");
        // Create Compute Pass
        if (!mpReducePass)
        {
            Program::Desc desc;
            desc.addShaderLibrary(kReduceShader).csEntry("main").setShaderModel("6_6");

            DefineList defines;
            defines.add("COUNT_LIGHTS", std::to_string(mLightCount));

            mpReducePass = ComputePass::create(mpDevice, desc, defines, true);
        }

        auto var = mpReducePass->getRootVar();

        const uint maxMipCount = mImportanceMap[0]->getMipCount() - 1u;
        for (uint mip = 0; mip < maxMipCount; mip += 5)
        {
            uint dstMip = math::min(maxMipCount, mip + 5);

            uint3 dispatchDim = uint3(mImportanceMap[0]->getWidth(mip), mImportanceMap[0]->getHeight(mip), mLightCount);

            dispatchDim.xy() = dispatchDim.xy() / 2u;

            var["CB"]["gDstSize"] = dispatchDim.xy();

            for (uint i = 0; i < mLightCount; i++)
            {
                var["gSrc"][i].setSrv(mImportanceMap[i]->getSRV(mip, 1u));
                var["gDst"][i].setUav(mImportanceMap[i]->getUAV(dstMip, 0u, 1u));
            }

            mpReducePass->execute(pRenderContext, dispatchDim);
        }
    }
}

void ImportanceMapHelper::distributeBudgetPass(
    RenderContext* pRenderContext,
    ref<Buffer>& pLastFrameNodeCounter,
    uint maxNodeBufferSize,
    uint maxRayBudget,
    bool oneBufferForAllLights
)
{
    FALCOR_PROFILE(pRenderContext, "DistributeSampleBudget");
    // Create Compute Pass
    if (!mpDistributeBudgetPass)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kDistributeBudgetShader).csEntry("main").setShaderModel("6_6");

        DefineList defines;
        defines.add("COUNT_LIGHTS", std::to_string(mLightCount));
        defines.add("MAX_RAY_SAMPLES", std::to_string(maxRayBudget));
        defines.add("USE_ONE_BUFFER_FOR_ALL_LIGHTS", oneBufferForAllLights ? "1" : "0");

        mpDistributeBudgetPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    //Update Runtime defines
    auto var = mpDistributeBudgetPass->getRootVar();
    mpDistributeBudgetPass->getProgram()->addDefine("MAX_RAY_SAMPLES", std::to_string(maxRayBudget));
    mpDistributeBudgetPass->getProgram()->addDefine("USE_ONE_BUFFER_FOR_ALL_LIGHTS", oneBufferForAllLights ? "1" : "0");

    // Determine Total Sample Budget
    if (mEnableDynamicRayCountCalc && !mResetRayCount)
    {
        // Get mip level
        uint mip = mSampleDistributions[0]->getMipCount() - 1;
        var["CB"]["gDetermineTotalSampleBudget"] = true;
        var["CB"]["gMaxNodeSize"] = int(maxNodeBufferSize * mDynRCGuardPercentage);
        var["CB"]["gChangePercentageIncrease"] = mDynRCChangePercentage.x;
        var["CB"]["gChangePercentageDecrease"] = mDynRCChangePercentage.y;

        var["gLastFrameRayBudget"] = mpLastFrameTotalRayBudget;
        for (uint i = 0; i < mLightCount; i++)
        {
            var["gImportance"][i].setSrv(mImportanceMap[i]->getSRV(mip, 1u));
            var["gSampleBudget"][i].setUav(mSampleDistributions[i]->getUAV(mip));
        }
       
        var["gNodeCount"] = pLastFrameNodeCounter;
        mpDistributeBudgetPass->execute(pRenderContext, uint3(1, 1, 1));
    }

    // Reset Total Sample Budget to base (Importance Map) resolution
    if (mResetRayCount)
    {
        for (uint i = 0; i < mLightCount; i++)
        {
            pRenderContext->clearUAV(
                mSampleDistributions[i]->getUAV(mSampleDistributions[i]->getMipCount() - 1).get(), float4(mResolution.x * mResolution.y)
            );
        }

        mResetRayCount = false;
    }

    // Compute Pass to distribute Total Sample Budget to Sample Budget Map using the Importance
    var["CB"]["gDetermineTotalSampleBudget"] = false;
    uint highestMip = mImportanceMap[0]->getMipCount() - 1;
    for (uint i = 0; i < mLightCount; i++)
    {
        var["gImportance"][i].setSrv(mImportanceMap[i]->getSRV(0, 1u));                     // Base Level
        var["gTotalImportance"][i].setSrv(mImportanceMap[i]->getSRV(highestMip, 1u));        // Highest MIP (1x1)
        var["gSampleBudget"][i].setUav(mSampleDistributions[i]->getUAV(0));                   // Base Level
        var["gTotalSampleBudget"][i].setSrv(mSampleDistributions[i]->getSRV(highestMip, 1u)); // Highest MIP (1x1)
    }

    uint3 dispatchDim = uint3(mImportanceMap[0]->getWidth(0), mImportanceMap[0]->getHeight(0), mLightCount);
    var["CB"]["gDstSize"] = dispatchDim.xy();

    mpDistributeBudgetPass->execute(pRenderContext, dispatchDim);
}

void ImportanceMapHelper::genSampleDistributionPass(RenderContext* pRenderContext, ref<Buffer>& pCurrentFrameNodeCounter)
{
    FALCOR_PROFILE(pRenderContext, "GenSampleDistribution");
    // Create Compute Pass
    if (!mpGenSampleDistributionPass)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kGenSampleDistributionShader).csEntry("main").setShaderModel("6_6");

        DefineList defines;
        defines.add("COUNT_LIGHTS", std::to_string(mLightCount));
        defines.add("CLEAR_COUNTER", mIsLinkedList ? "1" : "0");

        mpGenSampleDistributionPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    auto var = mpGenSampleDistributionPass->getRootVar();
    if (mIsLinkedList)
        var["gCounter"] = pCurrentFrameNodeCounter; 
    for (uint m = 1; m < mSampleDistributions[0]->getMipCount(); m++)
    {
        for (uint i = 0; i < mLightCount; i++)
        {
            var["gSampleDistribution0"][i].setUav(mSampleDistributions[i]->getUAV(m - 1, 0, 1));
            var["gSampleDistribution1"][i].setUav(mSampleDistributions[i]->getUAV(m, 0, 1));
        }
        uint3 dispatchDim = uint3(mImportanceMap[0]->getWidth(m), mImportanceMap[0]->getHeight(m), mLightCount);
        var["CB"]["gDispatchSize"] = dispatchDim.xy();

        mpGenSampleDistributionPass->execute(pRenderContext, dispatchDim);
    }
}

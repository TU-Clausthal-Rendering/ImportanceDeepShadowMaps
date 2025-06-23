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
#include "../DeepShadowMapMethod.h"

using namespace Falcor;

class IDSMMaskAndOpaqueShadowMap
{
public:
    virtual ~IDSMMaskAndOpaqueShadowMap() = default;
    IDSMMaskAndOpaqueShadowMap(ref<Device> pDevice, ref<Scene> pScene);

    enum class MaskGenerateMode : uint
    {
        Mask_ISM = 0,
        Mask_SM = 1,
        NoMask_SM = 2,
    };

    /** Generate resources needed to evaluate the Shadow Method (e.g. Shadow Map)
    * Should be called every frame and needs to be called before using any resources from that pass
    */
    void generate(
        RenderContext* pRenderContext,
        const RenderData& renderData,
        const DeepShadowMapMethod* pDeepShadowMapMethod,
        ref<SampleGenerator> pSampleGenerator,
        MaskGenerateMode genMode = MaskGenerateMode::Mask_ISM
    );

    //Get the layered mask texture
    ref<Texture> getMask() const { return mpMaskTex; }

    //Get layered mask shadow map
    std::vector<ref<Buffer>>& getMaskImportanceShadowMap() { return mMaskOpaqueImportanceShadowMaps; }

    //Regular shadow map
    ref<Texture> getMaskShadowMap() const { return mpMaskOpaqueShadowMap; }

    //Set blacklist status
    void enableBlacklist(bool enable) { mEnableBlacklistWithMaterialFlag = enable; }
    //Set mult factor
    void setIDMMultFactor(uint multFactor) { mOpaqueImportanceSMMultFactor = multFactor; }

private:
    void generateTransparencyMask(
        RenderContext* pRenderContext,
        const RenderData& renderData,
        const DeepShadowMapMethod* pDeepShadowMapMethod
    );

    void generateOpaqueMaskImportanceShadowMap(
        RenderContext* pRenderContext,
        const RenderData& renderData,
        const DeepShadowMapMethod* pDeepShadowMapMethod,
        ref<SampleGenerator> pSampleGenerator
    );

    void generateOpaqueMaskShadowMap(
        RenderContext* pRenderContext,
        const RenderData& renderData,
        const DeepShadowMapMethod* pDeepShadowMapMethod
    );

    //Constants
    const uint kMaxTemporal = 8;    //Max temporal accumulation (8bit)

    // Sync Resources
    static const uint kFramesInFlight = 3;       ///< Number of frames in flight for GPU/CPU sync
    static const uint kOverestimateDispatchConstant = 64; // 64 in each direction overestimate
    ref<GpuFence> mpFence;                       ///< Fence for CPU/GPU syncs
    std::array<uint64_t,kFramesInFlight> mFenceWaitValues;            // Fence values forCounter sync
    uint mStagingCount = 0;

    //Runtime
    ref<Device> mpDevice;
    ref<Scene> mpScene;
    uint mISMLastFrameBufferSize;   //<To rebuild buffer if size has changed

    uint mTemporalCounter = 0;  //< Current frame counter for the temporal mask 
    uint mFrameCount = 0;
    uint mOpaqueImportanceSMMultFactor = 4;     //1, 4 or 9 to work without any issues

    //Options
    bool mEnableOpaqueMaskShadowMaps = true;    //< Enables the opaque mask shadow map pass
    bool mEnableBlacklistWithMaterialFlag = false;  //< Enables blacklist with material flag (castShadows)
    bool mGenUseMaskToReject = true;            //< Enables / Disables mask to reject samples

    //Buffer and Textures
    ref<Texture> mpMaskRasterTex; //2D Array containing the masks for all shadow maps
    ref<Texture> mpMaskTex; //Containing the temporally accumulated shadow masks
    std::vector<ref<Buffer>> mMaskOpaqueImportanceShadowMaps;         //Importance shadow map with only opaque objects
    ref<Texture> mpMaskOpaqueShadowMap;          // Importance shadow map with only opaque objects
    ref<Sampler> mpMaskSampler;             //Mask sampler for the gen pass

    struct DispatchFeedback
    {
        ref<Buffer> gpu;
        ref<Buffer> cpu;
        std::vector<uint2> dispatchDims;
    };
    std::vector<DispatchFeedback> mDispatchFeedbackBuffers;

    // Pipelines / Programs
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

    RasterPipeline mGenerateMaskPip;
    ref<ComputePass> mpTemporalAccumulateMaskPass;
         
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
    RayTracingPipeline mGenerateMaskImportanceShadowMapRayPass;
    RayTracingPipeline mGenerateMaskShadowMapRayPass;
};

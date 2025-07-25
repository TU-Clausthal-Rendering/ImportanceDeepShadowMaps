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
#include "ErrorOverlayEnums.slang"

using namespace Falcor;

class ErrorOverlay : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ErrorOverlay, "ErrorOverlay", "Insert pass description here.");

    static ref<ErrorOverlay> create(ref<Device> pDevice, const Properties& props) { return make_ref<ErrorOverlay>(pDevice, props); }

    ErrorOverlay(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override {}
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;

private:
    void parseProperties(const Properties& props);
    void setOutputDirectory(const std::filesystem::path& path_);
    void captureCurrentFrame(const RenderData& renderData,const uint validImages,const uint4 imageTypes);

    //Runtime Vars
    uint mLineThickness = 1;
    float3 mLineColor = float3(1.0f);
    std::array<ErrorOverlayMode, 4> mImageModes = {
        ErrorOverlayMode::Image, ErrorOverlayMode::Reference, ErrorOverlayMode::None, ErrorOverlayMode::None};

    //Output
    std::string mBaseFilename = "Mogwai";
    std::filesystem::path mOutputDir = ".";
    bool mCaptureCurrentFrame = false;

    //Control Vars
    float2 mMousePos = float2(960.f,540.f);
    bool mMouseDown = false;
    int mOnlyShowID = -1;


    ref<ComputePass> mpErrorOverlayPass;
};

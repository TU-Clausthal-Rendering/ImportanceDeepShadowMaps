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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class SceneDebuggerMode : uint32_t
{
    // Geometry
    HitType,
    InstanceID,
    MaterialID,
    GeometryID,
    BlasID,
    InstancedGeometry,
    PrimitiveID,
    // Shading data
    FaceNormal,
    ShadingNormal,
    ShadingTangent,
    ShadingBitangent,
    FrontFacingFlag,
    BackfacingShadingNormal,
    TexCoords,
    // Material properties
    GuideNormal,
    Roughness,
    Metallic,
    Opacity,
    FlatShaded,
};

FALCOR_ENUM_INFO(SceneDebuggerMode, {
    // Geometry
    { SceneDebuggerMode::HitType, "HitType" },
    { SceneDebuggerMode::InstanceID, "InstanceID" },
    { SceneDebuggerMode::MaterialID, "MaterialID" },
    { SceneDebuggerMode::GeometryID, "GeometryID" },
    { SceneDebuggerMode::BlasID, "BlasID" },
    { SceneDebuggerMode::InstancedGeometry, "InstancedGeometry" },
    { SceneDebuggerMode::PrimitiveID, "PrimitiveID" },
    // Shading data
    { SceneDebuggerMode::FaceNormal, "FaceNormal" },
    { SceneDebuggerMode::ShadingNormal, "ShadingNormal" },
    { SceneDebuggerMode::ShadingTangent, "ShadingTangent" },
    { SceneDebuggerMode::ShadingBitangent, "ShadingBitangent" },
    { SceneDebuggerMode::FrontFacingFlag, "FrontFacingFlag" },
    { SceneDebuggerMode::BackfacingShadingNormal, "BackfacingShadingNormal" },
    { SceneDebuggerMode::TexCoords, "TexCoords" },
    // Material properties
    { SceneDebuggerMode::GuideNormal, "GuideNormal" },
    { SceneDebuggerMode::Roughness, "Roughness" },
    { SceneDebuggerMode::Metallic, "Metallic" },
    { SceneDebuggerMode::Opacity, "Opacity" },
    { SceneDebuggerMode::FlatShaded, "FlatShaded" },
});
FALCOR_ENUM_REGISTER(SceneDebuggerMode);

struct SceneDebuggerParams
{
    uint mode = (uint)SceneDebuggerMode::FaceNormal; ///< Current visualization mode. See SceneDebuggerMode.
    uint2 frameDim = { 0, 0 };
    uint frameCount = 0;

    uint2 selectedPixel = { 0, 0 }; ///< The currently selected pixel for readback.
    int flipSign = false;           ///< Flip sign before visualization.
    int remapRange = true;          ///< Remap valid range to [0,1] before output.
    int clamp = true;               ///< Clamp pixel values to [0,1] before output.

    int showVolumes = true;         ///< Show volumes.
    float densityScale = 1.f;       ///< Volume density scale factor.
    uint instanceMask = 0xFF;       ///< Ray Tracing Instance Mask
    uint _pad;  
};

struct PixelData
{
    static constexpr uint kInvalidID = 0xffffffff;

    // Geometry
    uint hitType;
    uint instanceID;
    uint materialID;
    uint geometryID;
    uint blasID;

    // ShadingData
    float3 posW;
    float3 V;
    float3 N;
    float3 T;
    float3 B;
    float2 uv;
    float3 faceN;
    float4 tangentW;
    uint frontFacing;
    float curveRadius;
};

enum class InstanceInfoFlags : uint32_t
{
    IsInstanced = 0x1,
};

struct InstanceInfo
{
    uint flags = 0;                 ///< Flags as a combination of 'InstanceInfoFlags' flags.
};

END_NAMESPACE_FALCOR

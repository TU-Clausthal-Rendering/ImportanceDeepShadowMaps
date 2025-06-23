#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct ParticleAnimateData {
    float lifetime;
    float3 velocity;
    bool isValid;
    uint _pad;
};

END_NAMESPACE_FALCOR

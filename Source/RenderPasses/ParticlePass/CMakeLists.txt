add_plugin(ParticlePass)

target_sources(ParticlePass PRIVATE
    ParticlePass.cpp
    ParticlePass.h
    UpdateParticlePoints.cs.slang
    ParticleDataTypes.slang
)

target_source_group(ParticlePass "RenderPasses")
target_copy_shaders(ParticlePass RenderPasses/ParticlePass)

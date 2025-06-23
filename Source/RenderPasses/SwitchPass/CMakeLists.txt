add_plugin(SwitchPass)

target_sources(SwitchPass PRIVATE
    SwitchPass.cpp
    SwitchPass.h
)

target_source_group(SwitchPass "RenderPasses")

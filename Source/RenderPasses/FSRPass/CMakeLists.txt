add_plugin(FSRPass)

target_sources(FSRPass PRIVATE
    FSRPass.cpp
    FSRPass.h
)

target_link_libraries(FSRPass PRIVATE ffxapi)

target_source_group(FSRPass "RenderPasses")

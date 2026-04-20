# Layer-type modules: attention, ffn, deltanet, ssm, moe.
# ffn added in Phase 1 (PR 1.5); deltanet, ssm, moe added in Phase 3.

add_library(qinf-layers STATIC
    attention.cpp
    ffn.cpp
)

target_include_directories(qinf-layers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:ggml,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(qinf-layers PUBLIC
    ggml
    qinf-state
)

set_target_properties(qinf-layers PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

function(make_app project_name file)
    set(NAME "${project_name}")

    add_executable(${NAME} "${file}")
    target_include_directories(${NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_SOURCE_DIR}/source/kibble"
        )

    set_target_properties(${NAME}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
            OUTPUT_NAME "${project_name}"
            EXCLUDE_FROM_ALL 1
        )

    target_link_libraries(${NAME}
        PRIVATE
            project_options
            project_warnings
            pthread
            stdc++fs
        )

endfunction()

add_custom_target(snippets)
# Loop over all cpp files in this directory
file(GLOB_RECURSE APP_SOURCES "*.cpp")
foreach(FILE ${APP_SOURCES})
    # Extract file name without extension and use this to declare a new target
    get_filename_component(APP_NAME ${FILE} NAME_WE)
    make_app(${APP_NAME} ${FILE})
    # Add dependency to the group target snippets,
    # then we can do "make snippets" to build all of them
    add_dependencies(snippets "${APP_NAME}")
    message(STATUS "Added example: ${APP_NAME}")
endforeach()
message(STATUS "All snippets can be built at once with 'make snippets'")

macro(register_app APP_NAME)
    file(GLOB APP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/app/${APP_NAME}/*.cu")
    file(GLOB APP_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/app/${APP_NAME}/*.h")

    add_executable(${APP_NAME} ${APP_SOURCES} ${APP_HEADERS})

    target_link_libraries(${APP_NAME} PUBLIC fluid_simulation_lib)
    target_include_directories(${APP_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/src")

    set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endmacro()

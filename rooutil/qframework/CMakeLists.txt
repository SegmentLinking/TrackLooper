#########################################################################################################
# Package: QFramework ###################################################################################

cmake_minimum_required(VERSION 2.8)
project( QFramework )
set(TQPATH ${PROJECT_SOURCE_DIR} PARENT_SCOPE)

IF(NOT WITHOUT_ASG)
  # register this as a package to ASG
  atlas_subdir( QFramework )
ENDIF()

# include libxml
find_package(LibXml2 REQUIRED)
include_directories(${LIBXML2_INCLUDE_DIR})
set(LIBS ${LIBS} ${LIBXML2_LIBRARIES})

# include ROOT
find_package( ROOT COMPONENTS Core Tree MathCore Hist RIO HistPainter Matrix Gpad Graf TreePlayer TMVA )

# set some variables for easier handling
set(QFrameworkLinkDef ${PROJECT_SOURCE_DIR}/Root/LinkDef.h)
set(QFrameworkPython ${PROJECT_SOURCE_DIR}/python/__init__.py)
file(GLOB QFrameworkSources Root/[A-Z]*.cxx)
file(GLOB QFrameworkHeaders QFramework/[A-Z]*.h)
file(GLOB QFrameworkPythonTests test/[A-Za-z]*.py)

#########################################################################################################
### ATTENTION: The auto-generated files are currently created at configuration time, not at build time! #

# generate the LinkDef.h and __init__.py
execute_process(
  COMMAND python ${PROJECT_SOURCE_DIR}/share/generateBindings.py --pkgname QFramework --verbose --python ${QFrameworkPython} --python-head  ${PROJECT_SOURCE_DIR}/python/QFramework.py --linkdef ${QFrameworkLinkDef} --headers ${QFrameworkHeaders}
)

# generate the definitions.h
execute_process(
  COMMAND python ${PROJECT_SOURCE_DIR}/share/generateLocals.py --verbose  --output ${PROJECT_SOURCE_DIR}/Root/definitions.h --set-working-directory ${PROJECT_SOURCE_DIR} --root --git --svn --gcc --set-named-path TQPATH ${PROJECT_SOURCE_DIR} --flag-header HAS_ROOLAGRANGIANMORPHING RooLagrangianMorphing.h --find-binary EXIFTOOLPATH exiftool --set-named-path LIBXMLPATH ${LIBXML2_LIBRARIES} --packages ${PACKAGES_STR}
)
# generate the locals.h
execute_process(
  COMMAND python ${PROJECT_SOURCE_DIR}/share/generateLocals.py --verbose  --output ${PROJECT_SOURCE_DIR}/Root/locals.h --set-working-directory ${PROJECT_SOURCE_DIR}
)
###   if one wanted to create them at build-time, something along the following lines would be needed   #

#add_custom_command(
#  OUTPUT ${QFrameworkLinkDef} 
#  DEPENDS ${QFrameworkHeaders}
#  COMMAND python ${PROJECT_SOURCE_DIR}/share/generateLinkDef.py --pkgname QFramework --output ${QFrameworkLinkDef} --headers ${QFrameworkHeaders}
#  COMMENT "generating LinkDef.h")

#########################################################################################################

if(NOT WITHOUT_ASG)
  # this section reflects the standard ASG way of configuring CMake
  # it is executed when compiling within an ASG environment
  find_package( GTest )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_XAOD")
  atlas_add_root_dictionary( QFramework QFrameworkCintDict
    ROOT_HEADERS ${QFrameworkHeaders} ${QFrameworkLinkDef}
    EXTERNAL_PACKAGES ROOT )
  atlas_add_library( QFramework
    ${QFrameworkHeaders} ${QFrameworkSources} ${QFrameworkCintDict}
    PUBLIC_HEADERS QFramework
    PRIVATE_INCLUDE_DIRS ${ROOT_INCLUDE_DIRS} 
    LINK_LIBRARIES xAODCore xAODRootAccess xAODTruth xAODBase xAODCutFlow 
    PRIVATE_LINK_LIBRARIES ${ROOT_LIBRARIES} xml2
    )
  atlas_install_python_modules( ${QFrameworkPython} )
  # register the test cases
  find_package(PythonInterp REQUIRED)
  foreach(TestScript ${QFrameworkPythonTests})
    get_filename_component(TestName ${TestScript} NAME_WE)
    atlas_add_test( ${TestName} SCRIPT ${TestScript} )
  endforeach()
ELSE()
  # this section is a standalone compilation segment
  # it is executed when compiling outside an ASG environment

  execute_process(
    COMMAND ln -sf ${QFrameworkPython} ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
    )
  file(APPEND ${CAFSETUP} "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${CMAKE_CURRENT_BINARY_DIR}/\n")
  
  # register all the files and directories
  include_directories ("${PROJECT_SOURCE_DIR}" "${ROOT_INCLUDE_DIRS}")

  # generate the dictionary source code
  ROOT_GENERATE_DICTIONARY(G__QFramework ${QFrameworkHeaders} LINKDEF ${QFrameworkLinkDef})

  # register the shared object to include both sources and dictionaries
  add_library( QFramework SHARED ${QFrameworkSources} G__QFramework.cxx)

  # link everything together at the end
  target_link_libraries( QFramework ${ROOT_LIBRARIES} )

  # register the test cases
  foreach(TestScript ${QFrameworkPythonTests})
    get_filename_component(TestName ${TestScript} NAME_WE)
    add_test( NAME ${TestName} COMMAND ${TestScript})
  endforeach()
ENDIF()

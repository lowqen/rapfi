# Android build (RapfiDroid)

This branch is the corresponding source of the engine shipped in RapfiDroid
(https://github.com/lowqen/RapfiDroid): upstream dhbloo/rapfi `3c94c2a9`, the
Android build changes in `Rapfi/CMakeLists.txt`, and three result-invariant
performance patches (K1–K3, one commit each). BENCH stays `e240ab16`.

RapfiDroid ships two executables, picked at run time by the CPU:

| File in the APK | CMake | PGO |
|---|---|---|
| `libengine.so` (any arm64) | `-DUSE_NEON=ON -DUSE_NEON_DOTPROD=OFF` | none |
| `libengine_dotprod.so` (ARMv8.2 dotprod) | `-DUSE_NEON=ON -DUSE_NEON_DOTPROD=ON` | `-fprofile-use=android/rapfi-dotprod.profdata` |

Common options: Android NDK r27c, `-DANDROID_ABI=arm64-v8a
-DANDROID_PLATFORM=android-26 -DANDROID_STL=c++_static
-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON -DCMAKE_BUILD_TYPE=Release
-DNO_COMMAND_MODULES=ON -DENABLE_LTO=ON`, then `llvm-strip --strip-all`.
For the dotprod build the profile flag goes into both `CMAKE_CXX_FLAGS` and
`CMAKE_EXE_LINKER_FLAGS` (LTO), with `-Wno-profile-instr-unprofiled`.

`rapfi-dotprod.profdata` is the merged LLVM profile the shipped dotprod build
was compiled with (collected on a Galaxy A16 by RapfiDroid `tools/pgo/train.sh`).
The full scripted build is RapfiDroid `tools/build_engine_android.sh`; the exact
binaries are described in its `app/src/main/jniLibs/ENGINE_VERSION.txt`.

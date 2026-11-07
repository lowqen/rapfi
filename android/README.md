# Android build (RapfiDroid)

This branch is the corresponding source of the engine shipped in RapfiDroid
(https://github.com/lowqen/RapfiDroid): upstream dhbloo/rapfi `3c94c2a9`, the
Android build changes in `Rapfi/CMakeLists.txt`, and result-invariant patches,
one commit each: K1–K3 and K7 (speed), R4 P2-2 (no weight reload when the thread
count changes), R4 P4-1 (periodic node counters, output only), and the database
save patches DB1 (atomic save through `<path>.tmp` + rename; saving no longer
stops or blocks the search), DB2 (`INFO USEDATABASE 1` keeps an attached database)
DB4 (the save runs on a background thread), DB5 (the yixindb records live in a
compact sorted table: same file bytes and answers, about a third of the memory)
and DB6 (`[database.yixindb] journal = true` saves only the changed records to
`<path>.journal`, folded into the file past `journal_compact_percent` or on
`YXCOMPACTDATABASE`). BENCH stays `e240ab16`.

The dotprod profile predates DB1–DB6; the compiler discards it only for the changed
database/protocol functions (none of them on the search path).

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

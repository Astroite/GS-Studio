#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "insta360_bridge.h"

int wmain(int argc, wchar_t** argv) {
  if (argc != 3) {
    std::wcerr << L"usage: gsstudio-media-helper.exe REQUEST.json RESPONSE.json\n";
    return 64;
  }
  try {
    return gsstudio::RunMediaRequest(std::filesystem::path(argv[1]),
                                 std::filesystem::path(argv[2]));
  } catch (const std::exception& error) {
    std::cerr << "MediaSDK helper failure: " << error.what() << "\n";
    return 70;
  }
}

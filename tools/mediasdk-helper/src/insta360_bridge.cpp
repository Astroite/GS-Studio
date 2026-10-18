#include "insta360_bridge.h"

#include <windows.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "json.h"
#include "metaData.h"
#include "stitcher/ins_stitcher.h"

namespace gsstudio {
namespace {

constexpr const char* kHelperVersion = "0.2.0";

// SDK path parameters are UTF-8 std::string (confirmed by the vendor sample's
// own BuildUtf8Args helper, which rebuilds argv from GetCommandLineW into
// UTF-8 specifically so non-ASCII paths work). This project's own paths are
// frequently non-ASCII (Chinese location/scene names), so every path pulled
// out of the request JSON is handed to the SDK exactly as parsed (UTF-8),
// with no codepage conversion.
//
// The C++ standard library's own filesystem/file-stream calls are a separate
// concern: on Windows, constructing std::filesystem::path from a narrow
// std::string goes through the *active code page*, not UTF-8, unless the
// string is first widened correctly. Everywhere this helper touches the
// filesystem itself (as opposed to handing a path to the SDK), it converts
// through Utf8ToWide/WideToUtf8 explicitly rather than relying on the
// implicit narrow-string constructor.
std::wstring Utf8ToWide(const std::string& text) {
  if (text.empty()) return std::wstring();
  int length = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                                    nullptr, 0);
  std::wstring wide(static_cast<size_t>(length), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
                       length);
  return wide;
}

std::string WideToUtf8(const std::wstring& wide) {
  if (wide.empty()) return std::string();
  int length = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                                    nullptr, 0, nullptr, nullptr);
  std::string out(static_cast<size_t>(length), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), out.data(), length,
                       nullptr, nullptr);
  return out;
}

std::filesystem::path Utf8Path(const std::string& utf8_path) {
  return std::filesystem::path(Utf8ToWide(utf8_path));
}

std::string ReadFileUtf8(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::string ModelRootDir() {
  const char* sdk_root = std::getenv("INSTA360_MEDIA_SDK_ROOT");
  if (sdk_root == nullptr || *sdk_root == '\0') {
    throw std::runtime_error("INSTA360_MEDIA_SDK_ROOT is not set");
  }
  // Verified against the actual extracted package layout: AI-Stitching /
  // ColorPlus / Denoise model files ship under bin/models, not a top-level
  // models/ directory.
  return std::string(sdk_root) + "\\bin\\models";
}

ins::STITCH_TYPE ParseStitchMode(const std::string& mode) {
  if (mode == "template") return ins::STITCH_TYPE::TEMPLATE;
  if (mode == "optflow") return ins::STITCH_TYPE::OPTFLOW;
  if (mode == "aiflow") return ins::STITCH_TYPE::AIFLOW;
  return ins::STITCH_TYPE::DYNAMICSTITCH;  // "dynamic" and any unrecognized value
}

ins::IMAGE_TYPE ParseImageType(const std::string& format) {
  if (format == "png") return ins::IMAGE_TYPE::PNG;
  return ins::IMAGE_TYPE::JPEG;  // "jpeg" and any unrecognized value
}

std::string FileExtensionLower(const std::string& utf8_path) {
  auto dot = utf8_path.find_last_of('.');
  if (dot == std::string::npos) return "";
  std::string ext = utf8_path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                  [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext;
}

// Camera models this build declares support for. This is GS Studio's own
// declaration, not something the SDK reports (GetMediaFileInfo/GetVersion
// carry no model list) -- mirrors sources.py's PUBLIC_SUPPORTED_CAMERAS
// exactly, since Python's gate requires a detected camera to appear in both
// its own set AND this response's supported_camera_models; a narrower list
// here would silently reject models Python would otherwise accept.
std::vector<std::string> SupportedCameraModels() {
  return {"Insta360 One X",  "Insta360 One R", "Insta360 One RS",  "Insta360 One X2",
          "Insta360 X3",     "Insta360 X4",    "Insta360 X4 Air", "Insta360 X5",
          "Insta360 X6"};
}

bool g_sdk_ready = false;

void EnsureSdkReady() {
  if (g_sdk_ready) return;
  ins::InitEnv();
  ins::SetModelFileRootDir(ModelRootDir());
  g_sdk_ready = true;
}

std::string SafeSdkVersion() { return g_sdk_ready ? ins::GetVersion() : "unknown"; }

struct ErrorInfo {
  std::string code;
  std::string message;
};

void WriteResponse(const std::filesystem::path& response_path, bool ok,
                    const std::string& sdk_version,
                    const std::vector<std::string>& supported_models,
                    const std::function<void(json::ObjectWriter&)>& extra_fields,
                    const ErrorInfo* error) {
  std::string out;
  {
    json::ObjectWriter writer(out);
    writer.Field("schema_version", static_cast<int64_t>(1));
    writer.Field("ok", ok);
    writer.Field("helper_version", kHelperVersion);
    writer.Field("sdk_version", sdk_version);
    writer.RawArray("supported_camera_models", [&] {
      for (size_t i = 0; i < supported_models.size(); ++i) {
        if (i > 0) out.push_back(',');
        json::WriteEscapedString(out, supported_models[i]);
      }
    });
    if (extra_fields) extra_fields(writer);
    if (error != nullptr) {
      writer.RawObject("error", [&](json::ObjectWriter& nested) {
        nested.Field("code", error->code);
        nested.Field("message", error->message);
      });
    }
  }
  std::ofstream response(response_path, std::ios::binary);
  if (!response) {
    throw std::runtime_error("cannot create response file");
  }
  response << out << "\n";
}

std::vector<std::string> ReadSourceFiles(const json::Value& request) {
  std::vector<std::string> files;
  for (const auto& item : request.at("source_files").as_array()) {
    files.push_back(item.as_string());
  }
  if (files.empty()) {
    throw std::runtime_error("source_files is empty");
  }
  return files;
}

void HandleCapabilities(const std::filesystem::path& response_path) {
  WriteResponse(response_path, /*ok=*/true, SafeSdkVersion(), SupportedCameraModels(), nullptr,
                nullptr);
}

void HandleProbe(const json::Value& request, const std::filesystem::path& response_path) {
  std::vector<std::string> source_files = ReadSourceFiles(request);

  ins::MediaFileInfo info{};
  if (!ins::GetMediaFileInfo(source_files, info)) {
    ErrorInfo error{"probe_failed", "MediaSDK could not parse the source file(s)"};
    WriteResponse(response_path, /*ok=*/false, SafeSdkVersion(), SupportedCameraModels(), nullptr,
                  &error);
    return;
  }

  ins_metadata::MetaDataParser metadata;
  std::string camera_model;
  if (metadata.Parse(source_files.front())) {
    camera_model = metadata.GetCameraType();
  }

  // MediaFileInfo has no frame_count field; derive it from fps * duration,
  // which is exact for constant-frame-rate Insta360 captures.
  int64_t frame_count = static_cast<int64_t>(
      std::llround(info.fps * static_cast<double>(info.duration_ms) / 1000.0));

  WriteResponse(
      response_path, /*ok=*/true, SafeSdkVersion(), SupportedCameraModels(),
      [&](json::ObjectWriter& writer) {
        writer.RawObject("media", [&](json::ObjectWriter& media) {
          media.Field("width", static_cast<int64_t>(info.width));
          media.Field("height", static_cast<int64_t>(info.height));
          media.Field("fps", info.fps);
          media.Field("frame_count", frame_count);
          media.Field("duration_seconds", static_cast<double>(info.duration_ms) / 1000.0);
          media.Field("codec", FileExtensionLower(source_files.front()));
          if (!camera_model.empty()) {
            media.Field("camera_model", camera_model);
          }
        });
      },
      nullptr);
}

struct StitchOutcome {
  bool succeeded = false;
  int error_code = 0;
  std::string error_message;
};

StitchOutcome RunStitchBlocking(ins::VideoStitcher& stitcher) {
  std::mutex mutex;
  std::condition_variable done_cv;
  bool done = false;
  StitchOutcome outcome;

  stitcher.SetStitchProgressCallback([&](int process, int error) {
    if (error == 0 && process < 100) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (done) return;
    done = true;
    outcome.succeeded = (error == 0);
    outcome.error_code = error;
    done_cv.notify_all();
  });
  stitcher.SetStitchStateCallback([&](int error, const char* errinfo) {
    if (error == 0) return;
    std::lock_guard<std::mutex> lock(mutex);
    if (done) return;
    done = true;
    outcome.succeeded = false;
    outcome.error_code = error;
    outcome.error_message = errinfo != nullptr ? errinfo : "";
    done_cv.notify_all();
  });

  stitcher.StartStitch();

  std::unique_lock<std::mutex> lock(mutex);
  // export_frames only stitches the requested still frames, not the whole
  // clip, but a large batch on a slow GPU is still given generous headroom
  // rather than a tight timeout that could fail a legitimate slow run.
  bool finished = done_cv.wait_for(lock, std::chrono::minutes(30), [&] { return done; });
  if (!finished) {
    stitcher.CancelStitch();
    outcome.succeeded = false;
    outcome.error_code = -1;
    outcome.error_message = "stitch timed out after 30 minutes";
  }
  return outcome;
}

void HandleExportFrames(const json::Value& request, const std::filesystem::path& response_path) {
  std::vector<std::string> source_files = ReadSourceFiles(request);
  const json::Array& frames = request.at("frames").as_array();
  if (frames.empty()) {
    throw std::runtime_error("frames is empty");
  }

  std::vector<uint64_t> frame_indices;
  std::vector<std::filesystem::path> requested_outputs;
  for (const auto& frame : frames) {
    frame_indices.push_back(static_cast<uint64_t>(frame.at("source_frame_index").as_int()));
    requested_outputs.push_back(Utf8Path(frame.at("output_file").as_string()));
  }

  const json::Value& output = request.at("output");
  const json::Value& stitching = request.at_or("stitching", json::Value(json::Object{}));

  // The SDK writes an auto-named image sequence into a directory; it does
  // not accept a per-frame filename template. Stage that sequence in a
  // scratch subdirectory next to the first requested output file (same
  // volume, same Windows-visible drive letter the caller already validated)
  // and rename each result onto its requested path afterward.
  std::filesystem::path scratch_dir =
      requested_outputs.front().parent_path() / L"gsdb-media-helper-scratch";
  std::filesystem::remove_all(scratch_dir);
  std::filesystem::create_directories(scratch_dir);

  ins::VideoStitcher stitcher;
  stitcher.SetInputPath(source_files);
  stitcher.EnableCuda(true);
  stitcher.SetStitchType(
      ParseStitchMode(stitching.at_or("mode", json::Value("dynamic")).as_string()));
  stitcher.EnableFlowState(stitching.at_or("flowstate", json::Value(false)).as_bool());
  stitcher.EnableDirectionLock(stitching.at_or("direction_lock", json::Value(false)).as_bool());
  stitcher.EnableDenoise(stitching.at_or("ai_denoise", json::Value(false)).as_bool());
  if (stitching.at_or("color_processing", json::Value(false)).as_bool()) {
    stitcher.EnableColorPlus(true);
  }
  if (stitching.at_or("sharpening", json::Value(false)).as_bool()) {
    // The protocol's "sharpening" is a boolean; the SDK's nearest control is
    // SetDefinition(0-100), a continuous strength with no dedicated on/off
    // primitive. 50 is a moderate, deliberately unexciting default.
    stitcher.SetDefinition(50);
  }

  int64_t out_width = output.at_or("width", json::Value(static_cast<int64_t>(0))).as_int();
  int64_t out_height = output.at_or("height", json::Value(static_cast<int64_t>(0))).as_int();
  if (out_width <= 0 || out_height <= 0) {
    // Despite SetOutputSize's header comment ("default size of source"), the
    // image-sequence export path does not actually fall back on its own --
    // confirmed empirically: omitting the call renders "invalid output
    // size: 0x0" and every frame fails with E_RENDER_FRAME. Resolve the
    // source's own resolution via GetMediaFileInfo and use that explicitly
    // whenever the request doesn't pin a specific size.
    ins::MediaFileInfo info{};
    if (!ins::GetMediaFileInfo(source_files, info)) {
      throw std::runtime_error("could not determine source resolution for output sizing");
    }
    out_width = info.width;
    out_height = info.height;
  }
  stitcher.SetOutputSize(static_cast<int>(out_width), static_cast<int>(out_height));

  stitcher.SetImageSequenceInfo(
      WideToUtf8(scratch_dir.wstring()),
      ParseImageType(output.at_or("format", json::Value("jpeg")).as_string()));
  stitcher.SetExportFrameSequence(frame_indices);

  StitchOutcome outcome = RunStitchBlocking(stitcher);
  if (!outcome.succeeded) {
    std::filesystem::remove_all(scratch_dir);
    ErrorInfo error{"stitch_failed",
                     outcome.error_message.empty()
                         ? ("MediaSDK stitch error " + std::to_string(outcome.error_code))
                         : outcome.error_message};
    WriteResponse(response_path, /*ok=*/false, SafeSdkVersion(), SupportedCameraModels(), nullptr,
                  &error);
    return;
  }

  // The SDK names sequence output by export order, not by source frame
  // index; sorting the produced files lexicographically recovers that order
  // (zero-padded sequence names sort correctly) and lines them back up with
  // `frames`, which was submitted in the same order as
  // SetExportFrameSequence. This ordering assumption is exactly what
  // scripts/test-mediasdk-helper.ps1's independent per-frame hash/dimension
  // re-check exists to catch if it turns out to be wrong.
  std::vector<std::filesystem::path> produced;
  for (const auto& entry : std::filesystem::directory_iterator(scratch_dir)) {
    if (entry.is_regular_file()) produced.push_back(entry.path());
  }
  std::sort(produced.begin(), produced.end());

  if (produced.size() != requested_outputs.size()) {
    std::filesystem::remove_all(scratch_dir);
    ErrorInfo error{"export_count_mismatch",
                     "MediaSDK produced " + std::to_string(produced.size()) +
                         " file(s) for " + std::to_string(requested_outputs.size()) +
                         " requested frame(s)"};
    WriteResponse(response_path, /*ok=*/false, SafeSdkVersion(), SupportedCameraModels(), nullptr,
                  &error);
    return;
  }

  for (size_t i = 0; i < produced.size(); ++i) {
    std::filesystem::create_directories(requested_outputs[i].parent_path());
    std::filesystem::rename(produced[i], requested_outputs[i]);
  }
  std::filesystem::remove_all(scratch_dir);

  WriteResponse(
      response_path, /*ok=*/true, SafeSdkVersion(), SupportedCameraModels(),
      [&](json::ObjectWriter& writer) {
        writer.Field("exported_count", static_cast<int64_t>(requested_outputs.size()));
      },
      nullptr);
}

}  // namespace

int RunMediaRequest(const std::filesystem::path& request_path,
                    const std::filesystem::path& response_path) {
  json::Value request;
  try {
    request = json::Parse(ReadFileUtf8(request_path));
  } catch (const std::exception& error) {
    // The request file itself could not be read or parsed: there is no SDK
    // version to report yet (EnsureSdkReady hasn't run), but the protocol
    // still requires a response file before a non-zero exit.
    ErrorInfo parse_error{"invalid_request", error.what()};
    WriteResponse(response_path, /*ok=*/false, "unknown", {}, nullptr, &parse_error);
    return 65;
  }

  try {
    EnsureSdkReady();
    std::string operation = request.at("operation").as_string();
    if (operation == "capabilities") {
      HandleCapabilities(response_path);
    } else if (operation == "probe") {
      HandleProbe(request, response_path);
    } else if (operation == "export_frames") {
      HandleExportFrames(request, response_path);
    } else {
      ErrorInfo error{"unknown_operation", "unrecognized operation: " + operation};
      WriteResponse(response_path, /*ok=*/false, SafeSdkVersion(), SupportedCameraModels(),
                    nullptr, &error);
      return 1;
    }
  } catch (const std::exception& error) {
    ErrorInfo failure{"helper_internal_error", error.what()};
    WriteResponse(response_path, /*ok=*/false, SafeSdkVersion(), SupportedCameraModels(), nullptr,
                  &failure);
    return 1;
  }
  return 0;
}

}  // namespace gsstudio

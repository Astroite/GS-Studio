#pragma once

#include <filesystem>

namespace gsstudio {

// Implements protocol v1 operations: capabilities, probe and export_frames.
// The approved SDK integration must call SetExportFrameSequence together with
// SetImageSequenceInfo for export_frames; whole-video frame dumps are forbidden.
int RunMediaRequest(const std::filesystem::path& request_path,
                    const std::filesystem::path& response_path);

}  // namespace gsstudio

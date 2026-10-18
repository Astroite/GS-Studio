# RealityScan registration export settings

`colmap-export-params.xml` is the checked-in RealityScan 2.2 registration-export
profile used by `gsstudio reconstruct`.

The profile exports a flat, ASCII COLMAP model without copying source images or
masks. The pipeline supplies a neutral export job filename (`registration.txt`);
RealityScan then writes the standard `cameras.txt`, `images.txt`, and
`points3D.txt` triplet. Do not use `images.txt` as the job filename: RealityScan
2.2 treats that reserved name ambiguously and can report success while omitting
the image/extrinsics member.

Camera calibration is not taken from this XML. The current Run calibration
policy allows ordinary perspective photos/video to estimate intrinsics, while
panorama-derived views retain their fixed projection intrinsics. The RealityScan
path currently uses an undistorted pinhole model. Preserve the actual exported
intrinsics rather than inferring them from this profile.

To replace the profile from the GUI, choose **Export Registration → COLMAP Text
Format**, keep **Export images**, **Export undistorted images**, and **Export
masks** disabled, choose the flat directory structure and ASCII file type, save
the settings, and review the resulting diff before committing it.

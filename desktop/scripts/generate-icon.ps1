$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class GsIconNative { [DllImport("user32.dll")] public static extern bool DestroyIcon(IntPtr handle); }'
$Bitmap = [Drawing.Bitmap]::new(256,256)
$Graphics = [Drawing.Graphics]::FromImage($Bitmap)
$Graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
$Graphics.Clear([Drawing.Color]::FromArgb(17,29,23))
$Graphics.TranslateTransform(128,128)
$Colors = @('#b4eb59','#55d1b6','#93b774')
for ($Index=0; $Index -lt 3; $Index++) {
    $Pen = [Drawing.Pen]::new([Drawing.ColorTranslator]::FromHtml($Colors[$Index]),7)
    $Graphics.DrawEllipse($Pen,-94,-49,188,98)
    $Graphics.RotateTransform(60)
    $Pen.Dispose()
}
$Brush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml('#b4eb59'))
$Graphics.FillEllipse($Brush,-18,-18,36,36)
$Brush.Dispose()
$Graphics.Dispose()
$Path = Join-Path $PSScriptRoot '../src-tauri/icons/icon.ico'
$Handle = $Bitmap.GetHicon()
$Icon = [Drawing.Icon]::FromHandle($Handle)
$Stream = [IO.File]::Create([IO.Path]::GetFullPath($Path))
try { $Icon.Save($Stream) } finally { $Stream.Dispose(); $Icon.Dispose(); [GsIconNative]::DestroyIcon($Handle) | Out-Null; $Bitmap.Dispose() }

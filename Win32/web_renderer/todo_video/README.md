FFMPEG version installed on my system and source code of codec dds

```
https://ffmpeg.org/doxygen/4.3/dds_8c_source.html
```

```
C:\Users\kdhome>ffmpeg --version
ffmpeg version 2024-07-07-git-0619138639-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg
developers
  built with gcc 13.2.0 (Rev5, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autod
etect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-bzli
b --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --
enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-sdl2 --enable-libaribb2
4 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc
 --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-libqrencode --enable-librav1e --enabl
e-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 -
-enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libvpx --enable-mediafound
ation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz
--enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda
-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-li
bvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libpl
acebo --enable-opencl --enable-libcdio --enable-libgme --enable-libmodplug --enable-libopenmpt --ena
ble-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame -
-enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-libopencore-amrn
b --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-li
bflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint
  libavutil      59. 28.100 / 59. 28.100
  libavcodec     61.  9.100 / 61.  9.100
  libavformat    61.  5.100 / 61.  5.100
  libavdevice    61.  2.100 / 61.  2.100
  libavfilter    10.  2.102 / 10.  2.102
  libswscale      8.  2.100 /  8.  2.100
  libswresample   5.  2.100 /  5.  2.100
  libpostproc    58.  2.100 / 58.  2.100
Unrecognized option '-version'.
Error splitting the argument list: Option not found
```

Convert Video Frames to PNG or BMP: First, use FFmpeg to extract frames from your video into a standard format like PNG or BMP.

```
ffmpeg -i input_video.mp4 -vf "fps=30" -pix_fmt rgba video_frame_%03d.png
// ffmpeg -i input_video.mp4 -vf "fps=30" -pix_fmt rgba video_frame_%03d.bmp
```

This craeted a lof of .png files
<br /><br />
Here's a simple Python script that will convert all PNG files in the current directory to DDS format:

```
import os
from PIL import Image

def convert_png_to_dds(png_filename):
    # Open the PNG image
    with Image.open(png_filename) as img:
        # Convert to RGBA if not already in that mode
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Save as DDS
        dds_filename = os.path.splitext(png_filename)[0] + '.dds'
        img.save(dds_filename, format='DDS')
        print(f"Converted {png_filename} to {dds_filename}")

def main():
    # Get all PNG files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.png'):
            convert_png_to_dds(filename)

if __name__ == "__main__":
    main()
```
Run
```
python convert_png_to_dds.py
```
This script create PNG file from ffmpeg to .DDS file. But this is a lot of space which need. Some gigabytes for 30 sec videos.
<br /> <br />

<b>Loading DDS Textures in Direct3D </b> <br />
Now that you have DDS files, you can modify your Direct3D code to load these textures. You can implement a loop to load each DDS texture for rendering frames sequentially.

Here's a conceptual outline of how you might load and display the textures in a loop:

```
#include <vector>
#include <string>

// Load all DDS frames into a vector
std::vector<ID3D11ShaderResourceView*> gTextureViews;
std::vector<std::wstring> gFrameFilenames;

void LoadAllTextures() {
    for (int i = 1; i <= numFrames; ++i) { // Assuming you know the number of frames
        std::wstring frameFilename = L"C:\\Windows\\Temp\\v2\\video_frame_";
        frameFilename += std::to_wstring(i);
        frameFilename += L".dds";
        
        ID3D11ShaderResourceView* textureView = nullptr;
        HRESULT hr = LoadTextureFromFile(frameFilename.c_str(), &textureView);
        if (SUCCEEDED(hr)) {
            gTextureViews.push_back(textureView);
        }
    }
}

void RenderFrame(int frameIndex) {
    // Bind the current frame texture to the pixel shader
    gDeviceContext->PSSetShaderResources(0, 1, &gTextureViews[frameIndex]);
    
    // Perform the rendering steps...
}

// In your main loop, you can loop through textures
int currentFrame = 0;
while (running) {
    RenderFrame(currentFrame);
    currentFrame = (currentFrame + 1) % gTextureViews.size(); // Loop through frames
    // Swap buffers, etc.
}
```

And in this "another_impl.cpp" file last two top implementations try to render this .DDS file as video. Frame by frame. But this is not working. This is trash code. But... 
<hr>
<b>Reverse from .dds / .png to .mp4</b> <br /><br />
Ensure Your Files are Sequentially Named: Make sure the .dds files are named in a sequential order, for example:

```
video_frame_001.dds
video_frame_002.dds
video_frame_003.dds
...
```
Option 1: Convert .dds to Another Format (e.g., PNG or BMP) <br />
Step 1: Convert .dds Files to .png <br />

```
import os
from PIL import Image

def convert_dds_to_png(dds_filename):
    # Open the DDS image
    with Image.open(dds_filename) as img:
        # Convert to PNG
        png_filename = os.path.splitext(dds_filename)[0] + '.png'
        img.save(png_filename, format='PNG')
        print(f"Converted {dds_filename} to {png_filename}")

def main():
    # Get all DDS files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.dds'):
            convert_dds_to_png(filename)

if __name__ == "__main__":
    main()
```

This script converts all .dds files in the current directory to .png.
<br /><br />
Step 2: Combine Converted PNG Files into a Video

```
ffmpeg -framerate 30 -i video_frame_%03d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
```

info --> FFmpeg doesn't support .dds files directly, so converting them first is necessary.
```
-framerate 30: Specifies the frame rate (adjust it based on your needs).
-i video_frame_%03d.png: Tells FFmpeg to use the image sequence (%03d specifies the sequence numbering format).
-c:v libx264: Specifies the H.264 codec for video compression.
-pix_fmt yuv420p: Sets the pixel format (required for compatibility with most players).
```
But first to png from dds. But in the first step at the top of this readme file as you see you do this. This was the first step, unpack .mp4 video file to .png frames (as images).

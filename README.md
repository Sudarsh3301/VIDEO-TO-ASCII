## 🚀 ** Usage Examples**

### Terminal with Color
```bash
# Colored ASCII in terminal
python enhanced_ascii.py video.mp4 --terminal-only --color

# Color + edge detection
python enhanced_ascii.py video.mp4 --terminal-only --color --edge canny

# Color + dithering
python enhanced_ascii.py video.mp4 --terminal-only --color --dithering floyd-steinberg

# All effects + color (BEST QUALITY)
python enhanced_ascii.py video.mp4 --terminal-only --color --edge hybrid --dithering atkinson
```

### Video Output with Color
```bash
# Create colored ASCII video file
python enhanced_ascii.py video.mp4 -o output.mp4 --color --edge hybrid --dithering floyd-steinberg
```

---

## 💡 **Recommendations**

**For best quality:**
```bash
python enhanced_ascii.py video.mp4 --terminal-only --color --edge hybrid --dithering atkinson -w 120
```

**For retro look:**
```bash
python enhanced_ascii.py video.mp4 --terminal-only --edge sobel --dithering atkinson -s retro
```

**For maximum detail:**
```bash
python enhanced_ascii.py video.mp4 --terminal-only --color --edge canny --dithering floyd-steinberg -w 150
```

## For `basic.py`


---

### ASCII Video Converter (`basic.py`)

This module converts video files into ASCII art videos while preserving the original audio. It provides multiple rendering styles and customization options.


#### Installation

Ensure you have the required dependencies:

```bash
pip install opencv-python numpy pillow moviepy
```

#### Usage

##### Command Line

```bash
# Basic usage
python basic.py input_video.mp4

# With custom output path
python basic.py input_video.mp4 -o output.mp4

# Custom ASCII width and style
python basic.py input_video.mp4 -w 80 -s detailed

# With color ASCII rendering
python basic.py input_video.mp4 --color

# Full customization
python basic.py input_video.mp4 -o output.mp4 -w 120 -s blocky -f 14 --color
```

##### Command Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `input` | - | string | - | Input video file path (required) |
| `--output` | `-o` | string | `ascii_output.mp4` | Output video file path |
| `--width` | `-w` | int | `100` | ASCII width in characters |
| `--style` | `-s` | choice | `standard` | ASCII style: `standard`, `detailed`, `blocky`, `retro` |
| `--font-size` | `-f` | int | `12` | Font size for ASCII rendering |
| `--color` | - | flag | False | Enable color ASCII rendering |


#### ASCII Styles

- **`standard`**: Basic ASCII characters - fastest, minimal file size
- **`detailed`**: Extended character set - more detail, slower processing
- **`blocky`**: Block characters (░▒▓█) - unique visual effect
- **`retro`**: Limited character set - retro computer aesthetic

#### Examples

```bash
# Convert with detailed ASCII style
python basic.py movie.mp4 -s detailed -w 120

# Create a blocky ASCII version with larger font
python basic.py video.mp4 -s blocky -f 16 -o blocky_output.mp4

# Generate color ASCII art video
python basic.py input.mp4 --color -w 80 -o colored.mp4
```

#### Output

The converter generates:
- **ASCII Video**: MP4 file with ASCII art rendering
- **Audio**: Original audio from source video preserved
- **Progress**: Real-time conversion progress displayed in console

#### Notes

- Conversion time depends on video length, resolution, and ASCII width
- Larger ASCII widths produce higher quality but slower conversion
- Color ASCII mode is significantly slower than standard mode
- Monospace fonts are automatically detected; fallback to default if unavailable


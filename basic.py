import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import subprocess

class ASCIIVideoConverter:
    def __init__(self, font_size=12, width=100, style="standard"):
        self.font_size = font_size
        self.width = width
        self.style = style
        
        # Try to load a monospace font
        try:
            # Common monospace fonts
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
                "C:/Windows/Fonts/consola.ttf",  # Windows
                "C:/Windows/Fonts/lucon.ttf",    # Windows
                "/System/Library/Fonts/Menlo.ttc",  # macOS
                "/Library/Fonts/Andale Mono.ttf"    # macOS
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, font_size)
                    break
            else:
                # Fallback to default font
                self.font = ImageFont.load_default()
                print("Using default font - install a monospace font for better results")
        except:
            self.font = ImageFont.load_default()
        
        # Character sets
        self.char_sets = {
            "standard": " .:-=+*#%@",
            "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
            "blocky": " ░▒▓█",
            "retro": " .oO8@",
        }
        
        self.ascii_chars = self.char_sets.get(style, self.char_sets["standard"])
    
    def frame_to_ascii_image(self, frame, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        """
        Convert frame to ASCII and render as PIL Image
        """
        # Calculate dimensions
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = int(self.width * aspect_ratio * 0.5)  # 0.5 for character aspect ratio
        height = max(1, height)
        
        # Resize frame
        resized = cv2.resize(frame, (self.width, height))
        
        # Convert to grayscale
        if len(resized.shape) > 2:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Normalize and map to ASCII
        normalized = gray / 255.0
        ascii_indices = (normalized * (len(self.ascii_chars) - 1)).astype(np.uint8)
        
        # Create ASCII text
        ascii_lines = []
        for row in ascii_indices:
            line = ''.join(self.ascii_chars[idx] for idx in row)
            ascii_lines.append(line)
        
        ascii_text = '\n'.join(ascii_lines)
        
        # Create image
        char_width = self.font_size * 0.6  # Approximate character width
        char_height = self.font_size * 1.2  # Approximate character height
        
        img_width = int(self.width * char_width)
        img_height = int(height * char_height)
        
        image = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(image)
        
        # Draw ASCII text
        draw.text((0, 0), ascii_text, font=self.font, fill=text_color)
        
        return image
    
    def convert_video_with_audio(self, input_path, output_path="ascii_video.mp4", 
                                 fps=None, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        """
        Convert video to ASCII video with original audio
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        video_fps = int(cap.get(cv2.CAP_PROP_FPS)) if fps is None else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get first frame to determine dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video")
            cap.release()
            return
        
        # Create first ASCII image to get output dimensions
        first_ascii_img = self.frame_to_ascii_image(first_frame, bg_color, text_color)
        output_width, output_height = first_ascii_img.size
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_ascii_video.mp4"
        
        out = cv2.VideoWriter(temp_video_path, fourcc, video_fps, 
                             (output_width, output_height))
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"Converting video to ASCII...")
        print(f"Output: {output_width}x{output_height} @ {video_fps} FPS")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to ASCII image
            ascii_img = self.frame_to_ascii_image(frame, bg_color, text_color)
            
            # Convert PIL Image to OpenCV format
            ascii_cv = cv2.cvtColor(np.array(ascii_img), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(ascii_cv)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release resources
        cap.release()
        out.release()
        
        print("\nAdding audio...")
        
        # Merge audio with video using moviepy
        try:
            # Load original video for audio
            original_clip = VideoFileClip(input_path)
            audio = original_clip.audio
            
            # Load ASCII video
            ascii_clip = VideoFileClip(temp_video_path)
            
            # Set audio to ASCII video
            final_clip = ascii_clip.set_audio(audio)
            
            # Write final video
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Close clips
            original_clip.close()
            ascii_clip.close()
            final_clip.close()
            
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            print(f"\n✅ Success! ASCII video with audio saved to: {output_path}")
            print(f"   Resolution: {output_width}x{output_height}")
            print(f"   Duration: {final_clip.duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error adding audio: {e}")
            print(f"ASCII video without audio saved to: {temp_video_path}")
    
    def convert_with_color_ascii(self, input_path, output_path="color_ascii_video.mp4"):
        """
        Create ASCII video with color approximation
        """
        cap = cv2.VideoCapture(input_path)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get dimensions from first frame
        ret, first_frame = cap.read()
        height = int(self.width * first_frame.shape[0] / first_frame.shape[1] * 0.5)
        height = max(1, height)
        
        char_width = self.font_size * 0.6
        char_height = self.font_size * 1.2
        
        output_width = int(self.width * char_width)
        output_height = int(height * char_height)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_color_ascii.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, video_fps, 
                             (output_width, output_height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("Creating color ASCII video...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized = cv2.resize(frame, (self.width, height))
            
            # Convert to color ASCII
            ascii_image = self._frame_to_color_ascii_image(resized)
            
            # Convert to OpenCV format
            ascii_cv = cv2.cvtColor(np.array(ascii_image), cv2.COLOR_RGB2BGR)
            out.write(ascii_cv)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        # Add audio
        self._add_audio_to_video(input_path, temp_video_path, output_path)
    
    def _frame_to_color_ascii_image(self, frame):
        """Convert frame to color ASCII image"""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize and get ASCII indices
        normalized = gray / 255.0
        ascii_indices = (normalized * (len(self.ascii_chars) - 1)).astype(np.uint8)
        
        # Create image
        char_width = self.font_size * 0.6
        char_height = self.font_size * 1.2
        
        img_width = int(width * char_width)
        img_height = int(height * char_height)
        
        image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw colored ASCII characters
        for y in range(height):
            for x in range(width):
                char = self.ascii_chars[ascii_indices[y, x]]
                # Get color from original frame
                b, g, r = frame[y, x]
                # Calculate position
                pos_x = int(x * char_width)
                pos_y = int(y * char_height)
                # Draw character
                draw.text((pos_x, pos_y), char, font=self.font, fill=(r, g, b))
        
        return image
    
    def _add_audio_to_video(self, source_video, ascii_video, output_path):
        """Merge audio from source video to ASCII video"""
        try:
            # Using ffmpeg directly (more reliable)
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', ascii_video,  # Input video
                '-i', source_video,  # Input audio
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',   # Audio codec
                '-map', '0:v:0', # Take video from first input
                '-map', '1:a:0', # Take audio from second input
                '-shortest',     # End with shortest stream
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Clean up
            if os.path.exists(ascii_video):
                os.remove(ascii_video)
            
            print(f"✅ Color ASCII video with audio saved to: {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
        except Exception as e:
            print(f"Error: {e}")

# Alternative: Simple function using only OpenCV and ffmpeg
def create_ascii_video_simple(input_path, output_path="ascii_output.mp4", width=80):
    """
    Simpler version using terminal-style rendering
    """
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Character set
    ascii_chars = " .:-=+*#%@"
    
    # Calculate output dimensions (using terminal character approximation)
    char_height_px = 20  # Approximate pixels per character row
    char_width_px = 10   # Approximate pixels per character
    
    output_width = width * char_width_px
    output_height = int(output_width * 9 / 16)  # 16:9 aspect
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = "temp_no_audio.mp4"
    out = cv2.VideoWriter(temp_video, fourcc, fps, (output_width, output_height))
    
    print("Processing video...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = int(width * aspect_ratio * 0.5)
        height = max(1, height)
        
        resized = cv2.resize(frame, (width, height))
        
        # Convert to grayscale and ASCII
        if len(resized.shape) > 2:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        normalized = gray / 255.0
        ascii_frame = ""
        
        for row in normalized:
            for pixel in row:
                index = int(pixel * (len(ascii_chars) - 1))
                ascii_frame += ascii_chars[index]
            ascii_frame += "\n"
        
        # Create blank image
        img = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Put ASCII text on image (simplified - would need proper font rendering)
        y = 30
        for line in ascii_frame.split('\n'):
            if line:
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.3, (255, 255, 255), 1, cv2.LINE_AA)
                y += 15
        
        out.write(img)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    cap.release()
    out.release()
    
    # Add audio using ffmpeg
    print("\nAdding audio...")
    add_audio_with_ffmpeg(input_path, temp_video, output_path)
    
    # Cleanup
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
    print(f"✅ Video saved to: {output_path}")

def add_audio_with_ffmpeg(source_video, video_no_audio, output_path):
    """Use ffmpeg to merge audio from source to new video"""
    try:
        import subprocess
        
        cmd = [
            'ffmpeg',
            '-y',
            '-i', video_no_audio,
            '-i', source_video,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
        else:
            print("Audio added successfully!")
            
    except Exception as e:
        print(f"Error using ffmpeg: {e}")
        print("Install ffmpeg: https://ffmpeg.org/download.html")

# Main function with command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert video to ASCII art video with audio')
    parser.add_argument('input', ,help='Input video file')
    parser.add_argument('-o', '--output', default='ascii_output.mp4', 
                       help='Output video file (default: ascii_output.mp4)')
    parser.add_argument('-w', '--width', type=int, default=100,
                       help='ASCII width in characters (default: 100)')
    parser.add_argument('-s', '--style', default='standard',
                       choices=['standard', 'detailed', 'blocky', 'retro'],
                       help='ASCII style (default: standard)')
    parser.add_argument('-f', '--font-size', type=int, default=12,
                       help='Font size for ASCII rendering (default: 12)')
    parser.add_argument('--color', action='store_true',
                       help='Use color ASCII (slower but prettier)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    print(f"Converting '{args.input}' to ASCII video...")
    print(f"Settings: width={args.width}, style={args.style}, font={args.font_size}")
    
    converter = ASCIIVideoConverter(
        font_size=args.font_size,
        width=args.width,
        style=args.style
    )
    
    if args.color:
        converter.convert_with_color_ascii(args.input, args.output)
    else:
        converter.convert_video_with_audio(args.input, args.output)

if __name__ == "__main__":
    main()
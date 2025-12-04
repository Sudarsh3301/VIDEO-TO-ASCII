import cv2
import numpy as np
import os
import time
import sys
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil

class EnhancedASCIIVideoConverter:
    def __init__(self, font_size=12, width=100, style="standard"):
        self.font_size = font_size
        self.width = width
        self.style = style
        
        # Try to load a monospace font
        try:
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
            "edge": " -|/\\+*#",  # Special set for edge detection
        }
        
        self.ascii_chars = self.char_sets.get(style, self.char_sets["standard"])
    
    def apply_floyd_steinberg_dithering(self, gray_frame):
        """Apply Floyd-Steinberg error diffusion dithering"""
        height, width = gray_frame.shape
        output = gray_frame.astype(float).copy()
        
        char_count = len(self.ascii_chars)
        
        for y in range(height):
            for x in range(width):
                old_pixel = output[y, x]
                new_pixel = round(old_pixel / 255.0 * (char_count - 1)) * (255.0 / (char_count - 1))
                output[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Diffuse error to neighboring pixels
                if x + 1 < width:
                    output[y, x + 1] += error * 7/16
                if y + 1 < height:
                    if x > 0:
                        output[y + 1, x - 1] += error * 3/16
                    output[y + 1, x] += error * 5/16
                    if x + 1 < width:
                        output[y + 1, x + 1] += error * 1/16
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def apply_atkinson_dithering(self, gray_frame):
        """Apply Atkinson dithering (lighter, more artistic)"""
        height, width = gray_frame.shape
        output = gray_frame.astype(float).copy()
        
        char_count = len(self.ascii_chars)
        
        for y in range(height):
            for x in range(width):
                old_pixel = output[y, x]
                new_pixel = round(old_pixel / 255.0 * (char_count - 1)) * (255.0 / (char_count - 1))
                output[y, x] = new_pixel
                
                error = (old_pixel - new_pixel) / 8.0
                
                # Atkinson pattern
                offsets = [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)]
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        output[ny, nx] += error
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def apply_edge_detection(self, frame, mode='canny'):
        """Apply edge detection and blend with original"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if mode == 'canny':
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            # Blend edges with original
            blended = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
            
        elif mode == 'sobel':
            # Sobel gradient
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            gradient = np.uint8(gradient / gradient.max() * 255)
            # Blend with original
            blended = cv2.addWeighted(gray, 0.6, gradient, 0.4, 0)
            
        elif mode == 'hybrid':
            # Combine both Canny and Sobel
            edges = cv2.Canny(gray, 50, 150)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            gradient = np.uint8(gradient / gradient.max() * 255)
            # Blend all three
            blended = cv2.addWeighted(gray, 0.5, edges, 0.25, 0)
            blended = cv2.addWeighted(blended, 1.0, gradient, 0.25, 0)
        
        else:
            blended = gray
        
        return blended
    
    def frame_to_ascii_text(self, frame, dithering=None, edge_mode=None):
        """Convert frame to plain ASCII text"""
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = max(1, int(self.width * aspect_ratio * 0.5))
        
        resized = cv2.resize(frame, (self.width, height))
        
        # Apply edge detection if requested (needs color/BGR frame)
        if edge_mode:
            # Make sure we have a color frame for edge detection
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = self.apply_edge_detection(resized, mode=edge_mode)
        else:
            # Convert to grayscale
            if len(resized.shape) > 2:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
        
        # Apply dithering if requested
        if dithering == 'floyd-steinberg':
            gray = self.apply_floyd_steinberg_dithering(gray)
        elif dithering == 'atkinson':
            gray = self.apply_atkinson_dithering(gray)
        
        # Normalize and map to ASCII
        normalized = gray / 255.0
        ascii_indices = (normalized * (len(self.ascii_chars) - 1)).astype(np.uint8)
        
        # Create ASCII text
        lines = []
        for row in ascii_indices:
            line = ''.join(self.ascii_chars[idx] for idx in row)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def frame_to_ascii_image(self, frame, bg_color=(0, 0, 0), text_color=(255, 255, 255),
                            dithering=None, edge_mode=None):
        """Convert frame to ASCII and render as PIL Image"""
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = int(self.width * aspect_ratio * 0.5)
        height = max(1, height)
        
        resized = cv2.resize(frame, (self.width, height))
        
        # Apply edge detection if requested (needs color/BGR frame)
        if edge_mode:
            # Make sure we have a color frame for edge detection
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = self.apply_edge_detection(resized, mode=edge_mode)
        else:
            # Convert to grayscale
            if len(resized.shape) > 2:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
        
        # Apply dithering if requested
        if dithering == 'floyd-steinberg':
            gray = self.apply_floyd_steinberg_dithering(gray)
        elif dithering == 'atkinson':
            gray = self.apply_atkinson_dithering(gray)
        
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
        char_width = self.font_size * 0.6
        char_height = self.font_size * 1.2
        
        img_width = int(self.width * char_width)
        img_height = int(height * char_height)
        
        image = Image.new('RGB', (img_width, img_height), color=bg_color)
        draw = ImageDraw.Draw(image)
        
        # Draw ASCII text
        draw.text((0, 0), ascii_text, font=self.font, fill=text_color)
        
        return image
    
    def play_in_terminal(self, input_path, fps=None, dithering=None, edge_mode=None, use_color=False):
        """Play ASCII video directly in terminal with audio"""
        import threading
        import pygame
        
        # Get terminal size
        term_width, term_height = shutil.get_terminal_size()
        
        cap = cv2.VideoCapture(input_path)
        video_fps = fps or int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = 1.0 / video_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Adjust ASCII width to terminal
        original_width = self.width
        self.width = min(self.width, term_width - 2)
        
        # Extract audio to permanent file
        audio_file = input_path.rsplit('.', 1)[0] + '_audio.mp3'
        audio_exists = os.path.exists(audio_file)
        
        print("\033[2J")  # Clear screen
        print(f"🎬 Playing ASCII video with audio... (Press Ctrl+C to stop)")
        print(f"Settings: {self.width} chars wide, {video_fps} FPS")
        if use_color:
            print(f"Color mode: ENABLED")
        if dithering:
            print(f"Dithering: {dithering}")
        if edge_mode:
            print(f"Edge detection: {edge_mode}")
        print("-" * 50)
        
        # Extract audio if not already extracted
        if not audio_exists:
            print("Extracting audio...")
            try:
                from moviepy.editor import VideoFileClip
                video_clip = VideoFileClip(input_path)
                
                if video_clip.audio is None:
                    print("⚠ No audio track found in video")
                    audio_file = None
                else:
                    video_clip.audio.write_audiofile(audio_file, logger=None, verbose=False)
                    print(f"✓ Audio extracted to: {audio_file}")
                
                video_clip.close()
            except Exception as e:
                print(f"⚠ Audio extraction error: {e}")
                audio_file = None
        else:
            print(f"✓ Using existing audio file: {audio_file}")
        
        # Initialize pygame and load audio
        audio_loaded = False
        if audio_file and os.path.exists(audio_file):
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                audio_loaded = True
                print("✓ Audio loaded successfully")
            except Exception as e:
                print(f"⚠ Audio loading error: {e}")
        
        print("\nStarting playback in 1 second...")
        time.sleep(1)
        
        # Start audio playback
        if audio_loaded:
            try:
                pygame.mixer.music.play()
            except Exception as e:
                print(f"⚠ Audio playback error: {e}")
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                loop_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to ASCII text (with or without color)
                if use_color:
                    ascii_text = self.frame_to_ascii_text_color(frame, dithering=dithering, edge_mode=edge_mode)
                else:
                    ascii_text = self.frame_to_ascii_text(frame, dithering=dithering, edge_mode=edge_mode)
                
                # Clear and redraw
                print("\033[H", end='')  # Move cursor to home
                print(ascii_text, end='', flush=True)
                
                # Show stats at bottom
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                audio_icon = "🔊" if audio_loaded else "🔇"
                print(f"\n\n{audio_icon} Frame: {frame_count}/{total_frames} | Progress: {progress:.1f}% | FPS: {current_fps:.1f}", end='')
                
                frame_count += 1
                
                # Maintain framerate
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_delay - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Playback stopped by user")
        finally:
            # Stop audio
            if audio_loaded:
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                except:
                    pass
            
            cap.release()
            self.width = original_width
            
            print("\033[2J\033[H")  # Clear screen
            print("✓ Playback complete!")
            if audio_file and os.path.exists(audio_file):
                print(f"✓ Audio file saved: {audio_file}")
    
    def frame_to_ascii_text_color(self, frame, dithering=None, edge_mode=None):
        """Convert frame to colored ASCII text using ANSI escape codes"""
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = max(1, int(self.width * aspect_ratio * 0.5))
        
        resized = cv2.resize(frame, (self.width, height))
        
        # Store original for color extraction
        color_frame = resized.copy()
        
        # Apply edge detection if requested (needs color/BGR frame)
        if edge_mode:
            # Make sure we have a color frame for edge detection
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = self.apply_edge_detection(resized, mode=edge_mode)
        else:
            # Convert to grayscale
            if len(resized.shape) > 2:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
        
        # Apply dithering if requested
        if dithering == 'floyd-steinberg':
            gray = self.apply_floyd_steinberg_dithering(gray)
        elif dithering == 'atkinson':
            gray = self.apply_atkinson_dithering(gray)
        
        # Normalize and map to ASCII
        normalized = gray / 255.0
        ascii_indices = (normalized * (len(self.ascii_chars) - 1)).astype(np.uint8)
        
        # Create colored ASCII text with ANSI codes
        lines = []
        for y in range(height):
            line = ''
            for x in range(self.width):
                char = self.ascii_chars[ascii_indices[y, x]]

                # Get RGB color from original frame
                b, g, r = color_frame[y, x]

                # Create ANSI color code (24-bit true color)
                color_code = f"\033[38;2;{r};{g};{b}m"
                line += f"{color_code}{char}"

            # Reset color at end of line
            line += "\033[0m"
            lines.append(line)
        
        return '\n'.join(lines)
    
    def convert_video_with_effects(self, input_path, output_path="ascii_video.mp4", 
                                   fps=None, bg_color=(0, 0, 0), text_color=(255, 255, 255),
                                   dithering=None, edge_mode=None, use_color=False):
        """Convert video to ASCII with edge detection, dithering, and color"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video properties
        video_fps = int(cap.get(cv2.CAP_PROP_FPS)) if fps is None else fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Converting: {input_path}")
        print(f"Settings:")
        print(f"  - Width: {self.width} characters")
        print(f"  - Style: {self.style}")
        print(f"  - Color mode: {'ENABLED' if use_color else 'Disabled'}")
        print(f"  - Dithering: {dithering or 'None'}")
        print(f"  - Edge Detection: {edge_mode or 'None'}")
        print(f"  - FPS: {video_fps}")
        print(f"{'='*60}\n")
        
        # Get first frame to determine dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video")
            cap.release()
            return
        
        # Create first ASCII image to get output dimensions
        if use_color:
            first_ascii_img = self.frame_to_ascii_image_color(first_frame, dithering, edge_mode)
        else:
            first_ascii_img = self.frame_to_ascii_image(first_frame, bg_color, text_color, 
                                                         dithering, edge_mode)
        output_width, output_height = first_ascii_img.size
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_ascii_video.mp4"
        
        out = cv2.VideoWriter(temp_video_path, fourcc, video_fps, 
                             (output_width, output_height))
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"Processing frames...")
        print(f"Output: {output_width}x{output_height} @ {video_fps} FPS\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to ASCII image with effects
            if use_color:
                ascii_img = self.frame_to_ascii_image_color(frame, dithering, edge_mode)
            else:
                ascii_img = self.frame_to_ascii_image(frame, bg_color, text_color,
                                                      dithering, edge_mode)
            
            # Convert PIL Image to OpenCV format
            ascii_cv = cv2.cvtColor(np.array(ascii_img), cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(ascii_cv)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                      f"FPS: {fps_actual:.1f} | ETA: {eta:.0f}s")
        
        # Release resources
        cap.release()
        out.release()
        
        print("\nAdding audio...")
        
        # Merge audio with video
        try:
            from moviepy.editor import VideoFileClip
            
            original_clip = VideoFileClip(input_path)
            audio = original_clip.audio
            
            ascii_clip = VideoFileClip(temp_video_path)
            final_clip = ascii_clip.set_audio(audio)
            
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None
            )
            
            original_clip.close()
            ascii_clip.close()
            final_clip.close()
            
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS!")
            print(f"{'='*60}")
            print(f"Output file: {output_path}")
            print(f"Resolution: {output_width}x{output_height}")
            print(f"Total frames: {total_frames}")
            print(f"Processing time: {time.time() - start_time:.1f}s")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"Error adding audio: {e}")
            print(f"ASCII video without audio saved to: {temp_video_path}")
    
    def frame_to_ascii_image_color(self, frame, dithering=None, edge_mode=None):
        """Convert frame to colored ASCII image"""
        aspect_ratio = frame.shape[0] / frame.shape[1]
        height = int(self.width * aspect_ratio * 0.5)
        height = max(1, height)
        
        resized = cv2.resize(frame, (self.width, height))
        
        # Store original for color extraction
        color_frame = resized.copy()
        
        # Apply edge detection if requested (needs color/BGR frame)
        if edge_mode:
            # Make sure we have a color frame for edge detection
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            gray = self.apply_edge_detection(resized, mode=edge_mode)
        else:
            # Convert to grayscale
            if len(resized.shape) > 2:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
        
        # Apply dithering if requested
        if dithering == 'floyd-steinberg':
            gray = self.apply_floyd_steinberg_dithering(gray)
        elif dithering == 'atkinson':
            gray = self.apply_atkinson_dithering(gray)
        
        # Normalize and map to ASCII
        normalized = gray / 255.0
        ascii_indices = (normalized * (len(self.ascii_chars) - 1)).astype(np.uint8)
        
        # Create image with colored characters
        char_width = int(self.font_size * 0.6)
        char_height = int(self.font_size * 1.2)
        
        img_width = self.width * char_width
        img_height = height * char_height
        
        image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Draw colored ASCII characters
        for y in range(height):
            for x in range(self.width):
                char = self.ascii_chars[ascii_indices[y, x]]
                
                # Get color from original frame
                b, g, r = color_frame[y, x]
                
                # Calculate position
                pos_x = x * char_width
                pos_y = y * char_height
                
                # Draw character with original color
                draw.text((pos_x, pos_y), char, font=self.font, fill=(int(r), int(g), int(b)))
        
        return image


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced ASCII Video Converter with Edge Detection & Dithering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python enhanced_ascii.py input.mp4
  
  # With edge detection
  python enhanced_ascii.py input.mp4 --edge canny
  
  # With dithering
  python enhanced_ascii.py input.mp4 --dithering floyd-steinberg
  
  # Combined effects
  python enhanced_ascii.py input.mp4 --edge hybrid --dithering atkinson
  
  # Terminal playback only (no file)
  python enhanced_ascii.py input.mp4 --terminal-only
  
  # Terminal playback with effects and COLOR
  python enhanced_ascii.py input.mp4 --terminal-only --edge sobel --dithering floyd-steinberg --color
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', default='ascii_output.mp4', 
                       help='Output video file (default: ascii_output.mp4)')
    parser.add_argument('-w', '--width', type=int, default=100,
                       help='ASCII width in characters (default: 100)')
    parser.add_argument('-s', '--style', default='standard',
                       choices=['standard', 'detailed', 'blocky', 'retro', 'edge'],
                       help='ASCII character set style (default: standard)')
    parser.add_argument('-f', '--font-size', type=int, default=12,
                       help='Font size for ASCII rendering (default: 12)')
    parser.add_argument('--dithering', choices=['floyd-steinberg', 'atkinson', 'none'],
                       help='Apply dithering algorithm for better quality')
    parser.add_argument('--edge', choices=['canny', 'sobel', 'hybrid'],
                       help='Apply edge detection (canny/sobel/hybrid)')
    parser.add_argument('--color', action='store_true',
                       help='Use colored ASCII (preserves original colors)')
    parser.add_argument('--terminal-only', action='store_true',
                       help='Play in terminal only, do not create video file')
    parser.add_argument('--fps', type=int, default=None,
                       help='Override output FPS (default: use source FPS)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Create converter
    converter = EnhancedASCIIVideoConverter(
        font_size=args.font_size,
        width=args.width,
        style=args.style
    )
    
    # Terminal-only mode
    if args.terminal_only:
        print("\n🎬 Terminal Playback Mode")
        converter.play_in_terminal(
            args.input, 
            fps=args.fps,
            dithering=args.dithering,
            edge_mode=args.edge,
            use_color=args.color
        )
    else:
        # Convert and save video
        converter.convert_video_with_effects(
            args.input,
            args.output,
            fps=args.fps,
            dithering=args.dithering,
            edge_mode=args.edge,
            use_color=args.color
        )


if __name__ == "__main__":
    main()
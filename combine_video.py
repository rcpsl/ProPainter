from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

def combine_videos(input_folder, output_file):
    # Get all mp4 files and sort them numerically
    video_files = [f for f in os.listdir(input_folder) if f.startswith('batch_') and f.endswith('.mp4')]
    video_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by batch number
    
    # Load all video clips
    clips = []
    for video_file in video_files:
        file_path = os.path.join(input_folder, video_file)
        print(f"Loading {video_file}...")
        clip = VideoFileClip(file_path)
        clips.append(clip)
    
    # Concatenate all clips
    print("Combining videos...")
    final_clip = concatenate_videoclips(clips)
    
    # Write the combined video to file
    print("Writing combined video to file...")
    final_clip.write_videofile(output_file, codec='libx264')
    
    # Close all clips to free up resources
    for clip in clips:
        clip.close()
    final_clip.close()
    
    print("Video combination complete!")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/siqili/ProPainter/results/rehab_preliminary_frames"  # Replace with your folder path
    output_file = "/home/siqili/ProPainter/results/rehab_preliminary_frames_combined.mp4"
    combine_videos(input_folder, output_file)
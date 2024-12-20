yt_dlp 2024.12.13 version - https://github.com/yt-dlp/yt-dlp/releases<br /><br />
Download a video example 

```
import yt_dlp

def list_formats(video_url):
    """
    List available formats for the given YouTube video URL.
    """
    ydl_opts = {
        'listformats': True,  # Option to list available formats
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return info

def download_video(video_url, max_resolution="1080p"):
    """
    Download the video with the highest resolution up to a specified max resolution.

    Args:
        video_url (str): The YouTube video URL.
        max_resolution (str): The maximum resolution allowed for download.
    """
    ydl_opts = {
        'format': f'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]',  # Filter formats
        'merge_output_format': 'mp4',  # Merge video and audio as MP4
        'outtmpl': '%(title)s.%(ext)s',  # Output file naming pattern
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# List and select format
video_link = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

# Display available formats
formats_info = list_formats(video_link)
print("\nAvailable formats:")
for f in formats_info['formats']:
   # print(f"{f['format_id']} - {f['ext']} - {f['resolution']} - {f['vcodec']} - {f['acodec']}")
   print(f"{f['format_id']} - {f['ext']} - {f['resolution']} - {f['vcodec']}")

for key in formats_info.keys():
    print(key)

for key in formats_info.keys():
    print(key, "\n", formats_info[key])

# Example: Only allow downloads up to 1080p
download_video(video_link, max_resolution="1080p")


"""
import yt_dlp

def download_video(video_url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',  # Ensure video is in MP4 format
        'merge_output_format': 'mp4',  # Merge video and audio as MP4
        'outtmpl': '%(title)s.%(ext)s',  # Output file naming pattern
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


#video_link = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
video_link = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
download_video(video_link)
"""
```

Download playlist example

```
import yt_dlp

def get_playlist_info(playlist_url):
    """
    Fetches and displays all video titles, lengths, and URLs from a playlist.

    Args:
        playlist_url (str): The URL of the YouTube playlist.

    Returns:
        list: A list of video URLs in the playlist.
    """
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
    
    print(playlist_info)
    
    
    video_links = []
    print("\nPlaylist Videos:")
    for i, video in enumerate(playlist_info['entries'], start=1):
        title = video.get('title', 'Unknown Title')
        duration = video.get('duration', 0)  # Duration in seconds
        video_url = video.get('original_url', 'Unknown URL')
        video_links.append(video_url)

        # Print video information
        print(f"{i}. Title: {title} | Duration: {duration // 60}m {duration % 60}s | URL: {video_url}")
    
    return video_links

def download_videos_from_array(video_links):
    """
    Downloads all videos from a list of video URLs in sequence.

    Args:
        video_links (list): A list of YouTube video URLs.
    """
    for idx, video_url in enumerate(video_links, start=1):
        print(f"\nDownloading video {idx}/{len(video_links)}: {video_url}")
        ydl_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]',
            'merge_output_format': 'mp4',
            'outtmpl': '%(title)s.%(ext)s',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

# Example usage
playlist_url = "https://www.youtube.com/playlist?list=PLTXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXyP5K0p" // change this to ID of playlist

# Get playlist info and links
video_links = get_playlist_info(playlist_url)

# Download all videos in sequence
download_videos_from_array(video_links)
```

"""
macOS Application Support path utilities for Video Indexer
Handles proper file locations according to macOS guidelines
"""
import os
import sys
from pathlib import Path

def get_app_support_dir():
    """Get the Application Support directory for Video Indexer"""
    if sys.platform == "darwin":  # macOS
        app_support = Path.home() / "Library" / "Application Support" / "Video Indexer"
    elif sys.platform == "win32":  # Windows
        app_support = Path(os.environ.get('APPDATA', Path.home())) / "Video Indexer"
    else:  # Linux and others
        app_support = Path.home() / ".video-indexer"
    
    # Create the directory if it doesn't exist
    app_support.mkdir(parents=True, exist_ok=True)
    return str(app_support)

def get_bundled_resource_path(resource_name):
    """Get path to a resource bundled with the app"""
    if getattr(sys, 'frozen', False):
        # Running in a bundle (py2app)
        bundle_dir = os.path.dirname(sys.executable)
        # Resources are in Contents/Resources/
        resource_path = os.path.join(bundle_dir, '..', 'Resources', resource_name)
        if os.path.exists(resource_path):
            return os.path.abspath(resource_path)
    
    # Running in development, look in current directory
    if os.path.exists(resource_name):
        return os.path.abspath(resource_name)
    
    return None

def get_data_dir(subdir_name):
    """Get a data directory within Application Support"""
    app_support = get_app_support_dir()
    data_dir = os.path.join(app_support, subdir_name)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

# Standard directories for the Video Indexer
def get_video_index_dir():
    """Get the video index output directory"""
    return get_data_dir("video_index_output")

def get_video_thumbnails_dir():
    """Get the video thumbnails directory"""
    return get_data_dir("video_thumbnails_output")

def get_temp_uploads_dir():
    """Get the temporary uploads directory"""
    return get_data_dir("temp_uploads")

def get_logs_dir():
    """Get the logs directory"""
    return get_data_dir("logs")

def is_running_as_app():
    """Check if running as a bundled app"""
    return getattr(sys, 'frozen', False)

def get_app_version():
    """Get app version info"""
    return "1.0.0"

if __name__ == "__main__":
    print("Video Indexer Path Configuration:")
    print(f"App Support Dir: {get_app_support_dir()}")
    print(f"Video Index Dir: {get_video_index_dir()}")
    print(f"Thumbnails Dir: {get_video_thumbnails_dir()}")
    print(f"Temp Uploads Dir: {get_temp_uploads_dir()}")
    print(f"Logs Dir: {get_logs_dir()}")
    print(f"Running as App: {is_running_as_app()}")
    print(f"App Version: {get_app_version()}") 
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import subprocess
import os
import sys
import threading
import webbrowser
import time
import queue # For thread-safe message passing
import re

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"
FRAME_EXTRACTOR_SCRIPT = "frame_extractor.py"
FRAME_INDEXER_SCRIPT = "frame_indexer.py"
SERVER_SCRIPT = "video_search_server_transcode.py"

# Import path utilities for proper macOS app support
try:
    from app_paths import get_video_index_dir, get_video_thumbnails_dir, get_temp_uploads_dir, get_bundled_resource_path, is_running_as_app
    OUTPUT_DIR = get_video_index_dir()
    print(f"Using Application Support directory: {OUTPUT_DIR}")
except ImportError:
    # Fallback for development
    OUTPUT_DIR = "video_index_output"
    print(f"Using development directory: {OUTPUT_DIR}")

SERVER_URL = "http://127.0.0.1:8002" # Default, ensure it matches server_script

class AppLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Indexer Launcher")
        self.root.geometry("900x550") # Wider to reduce text wrapping
        
        # Set window icon
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video-indexer.png")
            if os.path.exists(icon_path):
                self.root.iconphoto(True, tk.PhotoImage(file=icon_path))
        except Exception as e:
            print(f"Could not set window icon: {e}")  # Silent fallback

        self.video_directory = ""
        self.log_queue = queue.Queue() # Thread-safe queue for log messages
        self.server_process = None # To keep track of the server process
        self.active_processes = [] # Track all active processes for cleanup
        self.existing_index_detected = False
        
        # Progress tracking
        self.current_operation = ""
        self.progress_lines_to_skip = []
        
        # FP16 option
        self.use_fp16 = tk.BooleanVar()

        # Main workflow buttons (clean and focused)
        main_frame = tk.Frame(root)
        main_frame.pack(pady=15)

        # WORKAROUND: Use Labels as buttons since macOS ignores Button bg colors
        print("DEBUG: Creating Label-based buttons for guaranteed color visibility on macOS")
        
        def add_label_button_events(label, on_click, enabled_bg, disabled_bg, enabled_fg, disabled_fg, disabled_reason):
            # Store enabled/disabled state and reason
            label._enabled = True
            label._enabled_bg = enabled_bg
            label._disabled_bg = disabled_bg
            label._enabled_fg = enabled_fg
            label._disabled_fg = disabled_fg
            label._disabled_reason = disabled_reason
            
            def on_enter(e):
                if label._enabled:
                    label.config(bg='#2222AA')  # Hover color
            def on_leave(e):
                if label._enabled:
                    label.config(bg=label._enabled_bg)
            def on_press(e):
                if label._enabled:
                    on_click()
                else:
                    # Log reason for disabled state
                    self.log_message_to_queue(label._disabled_reason)
            label.bind("<Enter>", on_enter)
            label.bind("<Leave>", on_leave)
            label.bind("<Button-1>", on_press)
            # Helper to enable/disable
            def set_enabled(enabled):
                label._enabled = enabled
                if enabled:
                    label.config(bg=label._enabled_bg, fg=label._enabled_fg, cursor="hand2")
                else:
                    label.config(bg=label._disabled_bg, fg=label._disabled_fg, cursor="arrow")
            label.set_enabled = set_enabled
            return label
        
        self.select_dir_button = tk.Label(main_frame, text="📁 1. Select Video Directory", 
                                        width=25, height=2, bg="#000000", fg="#FFFFFF", font=("Arial", 10, "bold"),
                                        relief=tk.RAISED, bd=2, cursor="hand2")
        add_label_button_events(
            self.select_dir_button,
            self.select_video_directory,
            enabled_bg="#000000", disabled_bg="#555555", enabled_fg="#FFFFFF", disabled_fg="#AAAAAA",
            disabled_reason="Select directory is currently disabled. Wait for current operation to finish."
        )
        self.select_dir_button.pack(side=tk.LEFT, padx=5)

        self.start_button = tk.Label(main_frame, text="🚀 2. Start Processing & Server", 
                                   width=25, height=2, bg="#0000FF", fg="#FFFFFF", font=("Arial", 10, "bold"),
                                   relief=tk.RAISED, bd=2, cursor="hand2")
        add_label_button_events(
            self.start_button,
            self.start_full_process,
            enabled_bg="#0000FF", disabled_bg="#555555", enabled_fg="#FFFFFF", disabled_fg="#AAAAAA",
            disabled_reason="Start is currently disabled. Wait for current operation to finish or select a directory."
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        print("DEBUG: Label-based buttons created - these WILL show custom colors")

        # Optional utility (less prominent, shows only when relevant)
        self.utility_frame = tk.Frame(root, bg=root.cget('bg'))
        self.utility_frame.pack(pady=8, padx=10, fill=tk.X)
        
        self.clear_index_button = tk.Label(self.utility_frame, text="🧹 Clear All Indexes", 
                                          width=20, bg="#8B4513", fg="#FFFFFF", relief=tk.RAISED, bd=2, 
                                          cursor="hand2", font=("Arial", 9, "bold"))
        self.clear_index_button.bind("<Button-1>", lambda e: self.clear_index())
        self.clear_index_button.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.utility_frame, text="← Use if you need to start completely fresh", 
                fg="#888", font=("Arial", 9), bg=root.cget('bg')).pack(side=tk.LEFT, padx=10)
        
        # FP16 option frame
        fp16_frame = tk.Frame(root, bg=root.cget('bg'))
        fp16_frame.pack(pady=8, padx=10, fill=tk.X)
        
        fp16_checkbox = tk.Checkbutton(fp16_frame, text="🔧 Use FP16 precision (reduces memory usage ~50%)", 
                                      variable=self.use_fp16, font=("Arial", 9, "bold"),
                                      bg=root.cget('bg'), activebackground=root.cget('bg'))
        fp16_checkbox.pack(side=tk.LEFT)
        
        fp16_info_label = tk.Label(fp16_frame, text="← Recommended for CUDA GPUs, may be slower on Apple Silicon", 
                                  fg="#888", font=("Arial", 9), bg=root.cget('bg'))
        fp16_info_label.pack(side=tk.LEFT, padx=10)

        # Server status frame (initially hidden)
        self.server_status_frame = tk.Frame(root, bg=root.cget('bg'))
        self.server_status_label = tk.Label(self.server_status_frame, text="🟢 Server: Running at http://localhost:8002", 
                                           fg="#4CAF50", font=("Arial", 11, "bold"), bg=root.cget('bg'))
        self.server_status_label.pack(pady=8)

        # Management tools frame (initially hidden) - subtle dark styling
        self.management_frame = tk.Frame(root, bg=root.cget('bg'))
        
        # Subtle separator line
        separator = tk.Frame(self.management_frame, height=1, bg="#555", relief=tk.FLAT)
        separator.pack(fill=tk.X, pady=(0, 8))
        
        mgmt_title = tk.Label(self.management_frame, text="⚙️ Management Tools", 
                             font=("Arial", 9), bg=root.cget('bg'), fg="#888")
        mgmt_title.pack(pady=(0, 8))
        
        mgmt_button_frame = tk.Frame(self.management_frame, bg=root.cget('bg'))
        mgmt_button_frame.pack(pady=(0, 8))

        self.kill_servers_button = tk.Button(mgmt_button_frame, text="🔪 Kill Old Servers", command=self.kill_old_servers, 
                                           width=18, bg="white", fg="black", relief=tk.RAISED, 
                                           activebackground="#f0f0f0", font=("Arial", 9, "bold"))
        self.kill_servers_button.pack(side=tk.LEFT, padx=3)

        self.stop_server_button = tk.Button(mgmt_button_frame, text="🛑 Stop Server", command=self.stop_current_server, 
                                          width=15, bg="white", fg="black", relief=tk.RAISED,
                                          activebackground="#f0f0f0", font=("Arial", 9, "bold"))
        self.stop_server_button.pack(side=tk.LEFT, padx=3)

        self.stop_all_button = tk.Button(mgmt_button_frame, text="⏹️ Stop All", command=self.stop_all_processes, 
                                       width=12, bg="white", fg="black", relief=tk.RAISED,
                                       activebackground="#f0f0f0", font=("Arial", 9, "bold"))
        self.stop_all_button.pack(side=tk.LEFT, padx=3)
        
        # Initially hide server status and management tools
        self.server_status_frame.pack_forget()
        self.management_frame.pack_forget()

        # Label to show selected directory
        self.dir_label_var = tk.StringVar()
        self.dir_label_var.set("No directory selected.")
        self.dir_label = tk.Label(root, textvariable=self.dir_label_var, wraplength=880)
        self.dir_label.pack(pady=5)
        
        # Progress bar frame
        progress_frame = tk.Frame(root)
        progress_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.progress_label = tk.Label(progress_frame, text="Ready to start")
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        # ScrolledText widget for output
        self.output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, width=100)
        self.output_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.output_area.configure(state='disabled') # Make it read-only initially

        # Ensure the script runs from its own directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)
        self.log_message_to_queue(f"Running in directory: {self.script_dir}") # Use queue logger
        
        # Determine the correct Python executable (preferring virtual environment)
        self.python_executable = self._get_python_executable()
        self.log_message_to_queue(f"Using Python executable: {self.python_executable}")

        # Check for existing index and update UI accordingly
        self._check_existing_index()

        # Start processing the log queue periodically
        self.process_log_queue()

        # Set up window close handler to clean up server
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Attempt to bring to front if on macOS. Call this after UI elements are packed.
        if sys.platform == "darwin":
            self.root.after(200, self.bring_to_front_mac) # Slightly longer delay

    def bring_to_front_mac(self):
        if sys.platform == "darwin":
            self.log_message_to_queue("Attempting to bring application to front (macOS)...")
            # Try osascript (best effort, suppress most errors)
            try:
                script = '''tell application "System Events"
                            tell process "Python"
                                set frontmost to true
                            end tell
                         end tell'''
                subprocess.run(["osascript", "-e", script], capture_output=True, timeout=2, check=False)
            except Exception:
                pass # Intentionally suppress osascript errors if primary Tkinter methods work
            
            try:
                fallback_script = '''tell app "Python" to activate'''
                subprocess.run(["osascript", "-e", fallback_script], capture_output=True, timeout=2, check=False)
            except Exception:
                pass # Intentionally suppress osascript errors

            # Primary Tkinter methods for focus
            try:
                self.root.lift()
                self.root.attributes('-topmost', True)
                self.root.after(50, lambda: self.root.attributes('-topmost', False))
                self.root.focus_force()
                self.log_message_to_queue("Tkinter focus methods applied.")
            except tk.TclError as e:
                 self.log_message_to_queue(f"Note: TclError during Tkinter focus methods (window may be closing): {e}")
            except Exception as e_tk_focus:
                 self.log_message_to_queue(f"Note: Error during Tkinter focus methods: {e_tk_focus}")

    def log_message_to_queue(self, message):
        print(message) # Keep console logging for debug
        self.log_queue.put(message)
    
    def update_progress(self, operation, percentage):
        """Update the progress bar and label"""
        self.progress_label.config(text=f"{operation}: {percentage}%")
        self.progress_bar['value'] = percentage
        
    def parse_progress_message(self, message):
        """Parse progress messages and return (should_log, operation, percentage)"""
        import re
        
        # Look for various progress patterns
        patterns = [
            r'Progress:\s*(\d+)%',  # "Progress: 50%"
            r'(\d+)%\|',           # "50%|████████"
            r'(\d+)/(\d+)',        # "50/100" - convert to percentage
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                if pattern.endswith(r'(\d+)/(\d+)'):  # Fraction format
                    current, total = map(int, match.groups())
                    percentage = int((current / total) * 100) if total > 0 else 0
                else:
                    percentage = int(match.group(1))
                
                # Don't log repetitive progress lines
                if self.current_operation and percentage > 0:
                    return False, self.current_operation, percentage
                    
        return True, None, None

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                
                # Parse for progress information
                should_log, operation, percentage = self.parse_progress_message(message)
                
                if operation and percentage is not None:
                    self.update_progress(operation, percentage)
                
                # Only log the message if it's not a repetitive progress line
                if should_log:
                    self.output_area.configure(state='normal')
                    self.output_area.insert(tk.END, message + "\n")
                    self.output_area.see(tk.END)
                    self.output_area.configure(state='disabled')
                    
        except queue.Empty:
            pass # Should not happen with not self.log_queue.empty() check
        finally:
            # Reschedule self to run again after 100ms
            self.root.after(100, self.process_log_queue)

    def select_video_directory(self):
        selected_dir = filedialog.askdirectory(title="Select Folder Containing Your Videos")
        if selected_dir:
            self.video_directory = selected_dir
            self.dir_label_var.set(f"Selected: {self.video_directory}")
            self.existing_index_detected = False
            self.start_button.config(text="2. Start Processing & Server", bg="#0000FF", fg="#FFFFFF", font=("Arial", 10, "bold"))
            self.start_button.set_enabled(True)
            self.select_dir_button.config(text="1. Select Video Directory", bg="#000000", fg="#FFFFFF", font=("Arial", 10, "bold"))
            self.select_dir_button.set_enabled(True)
            self.management_frame.pack_forget()
            self.log_message_to_queue(f"Video directory selected: {self.video_directory}")
            self.log_message_to_queue("📋 Ready to process new video directory")
        else:
            if not self.existing_index_detected:
                self.dir_label_var.set("No directory selected.")
                self.start_button.set_enabled(False)
            self.log_message_to_queue("Video directory selection cancelled.")

    def clear_index(self):
        """Clear all indexed data from both local and Application Support directories"""
        result = messagebox.askyesno(
            "Clear All Indexes", 
            "This will permanently delete ALL indexed data including:\n\n"
            "• Video thumbnails\n"
            "• Frame images and indexes\n" 
            "• Transcript data and indexes\n"
            "• All metadata and manifests\n"
            "• All processing logs\n\n"
            f"From both local directory and:\n{OUTPUT_DIR}\n\n"
            "Are you sure you want to continue?"
        )
        
        if not result:
            self.log_message_to_queue("Index clearing cancelled by user.")
            return
            
        # Disable buttons during clearing (Labels use visual feedback instead of state)
        self.clear_index_button.config(bg="#555555", fg="#AAAAAA")
        self.clear_index_button.unbind("<Button-1>")
        self.select_dir_button.config(bg="#555555", fg="#AAAAAA") 
        self.select_dir_button.unbind("<Button-1>")
        self.start_button.config(bg="#555555", fg="#AAAAAA")
        self.start_button.unbind("<Button-1>")
        
        # Clear output area
        self.output_area.configure(state='normal')
        self.output_area.delete(1.0, tk.END)
        self.output_area.configure(state='disabled')
        
        # Run clearing in a thread
        def clear_thread():
            self.log_message_to_queue("🧹 Starting comprehensive index clearing...")
            
            # First, clear local directories using the script
            clear_script_path = os.path.join(self.script_dir, "clear_index.sh")
            if os.path.exists(clear_script_path):
                self.log_message_to_queue("📁 Clearing local directories...")
                success = self.run_command(["bash", clear_script_path], "Clearing local indexes")
                if not success:
                    self.log_message_to_queue("⚠️ Warning: Failed to clear some local directories")
            else:
                self.log_message_to_queue("⚠️ Warning: clear_index.sh not found, clearing manually...")
            
            # Then, clear Application Support directory manually
            self.log_message_to_queue(f"📁 Clearing Application Support directory: {OUTPUT_DIR}")
            try:
                # Import thumbnails directory
                try:
                    from app_paths import get_video_thumbnails_dir
                    thumbnails_dir = get_video_thumbnails_dir()
                except ImportError:
                    thumbnails_dir = "video_thumbnails_output"
                
                # Clear thumbnails
                if os.path.exists(thumbnails_dir):
                    self.log_message_to_queue(f"  🖼️ Clearing thumbnails: {thumbnails_dir}")
                    import shutil
                    shutil.rmtree(thumbnails_dir, ignore_errors=True)
                    os.makedirs(thumbnails_dir, exist_ok=True)
                
                # Clear main index directory
                if os.path.exists(OUTPUT_DIR):
                    self.log_message_to_queue(f"  📊 Clearing index data: {OUTPUT_DIR}")
                    # Remove specific files and subdirectories
                    items_to_remove = [
                        "video_index.faiss",
                        "video_frame_metadata.json", 
                        "index_config.json",
                        "extraction_manifest.json",
                        "transcript_index.faiss",
                        "transcript_metadata.json",
                        "transcript_index_config.json", 
                        "transcript_manifest.json",
                        "frame_images",
                        "transcripts",
                        "indexer_logs",
                        "extractor_logs",
                        "transcript_indexer_logs"
                    ]
                    
                    for item in items_to_remove:
                        item_path = os.path.join(OUTPUT_DIR, item)
                        if os.path.exists(item_path):
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path, ignore_errors=True)
                                self.log_message_to_queue(f"    ✅ Cleared directory: {item}")
                            else:
                                os.remove(item_path)
                                self.log_message_to_queue(f"    ✅ Removed file: {item}")
                
                self.log_message_to_queue("✅ Application Support directory cleared successfully")
                
                self.root.after(0, lambda: messagebox.showinfo("Success", "All indexes cleared successfully!\n\nYou can now select a video directory and start fresh indexing."))
                # Reset video directory selection since data is cleared
                self.video_directory = ""
                self.dir_label_var.set("No directory selected.")
                # Reset index detection state and UI
                self.existing_index_detected = False
                self.root.after(0, self._setup_new_index_ui)
                
            except Exception as e:
                self.log_message_to_queue(f"❌ Error clearing Application Support directory: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to clear some indexes: {e}\nCheck the log for details."))
            
            self.root.after(0, self._enable_buttons_after_clear)
        
        thread = threading.Thread(target=clear_thread)
        thread.daemon = True
        thread.start()
    
    def _enable_buttons_after_clear(self):
        """Re-enable buttons after clear operation"""
        # Restore original colors and functionality for Labels
        self.clear_index_button.config(bg="#8B4513", fg="#FFFFFF")
        self.clear_index_button.bind("<Button-1>", lambda e: self.clear_index())
        
        self.select_dir_button.set_enabled(True)
        self.start_button.set_enabled(bool(self.video_directory))

    def _get_python_executable(self):
        """Get the correct Python executable, preferring virtual environment"""
        # Check for virtual environment in the same directory
        venv_path = os.path.join(self.script_dir, "video-indexer")
        venv_python = os.path.join(venv_path, "bin", "python")
        
        if os.path.exists(venv_python):
            self.log_message_to_queue(f"🐍 Found virtual environment at: {venv_path}")
            return venv_python
        # else:
        #     self.log_message_to_queue(f"⚠️  Virtual environment not found at {venv_path}")
        #     self.log_message_to_queue(f"💡 Creating virtual environment for isolated dependencies...")
            
        #     # Try to create virtual environment automatically
        #     if self._create_virtual_environment(venv_path):
        #         self.log_message_to_queue(f"✅ Virtual environment created successfully")
        #         return venv_python
        #     else:
        #         self.log_message_to_queue(f"❌ Failed to create virtual environment")
        #         self.log_message_to_queue(f"   Using system Python: {sys.executable}")
        #         return sys.executable

    def _create_virtual_environment(self, venv_path):
        """Create virtual environment automatically for end users"""
        try:
            self.log_message_to_queue(f"🔧 Creating virtual environment at: {venv_path}")
            
            # Use the venv module to create virtual environment
            import venv
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(venv_path)
            
            self.log_message_to_queue(f"📦 Virtual environment created with pip")
            return True
            
        except Exception as e:
            self.log_message_to_queue(f"❌ Error creating virtual environment: {e}")
            self.log_message_to_queue(f"💭 You may need to install python3-venv package")
            return False

    def _check_existing_index(self):
        """Check if there's an existing index and update UI accordingly"""
        # Check for key index files
        index_files = [
            os.path.join(OUTPUT_DIR, "video_index.faiss"),
            os.path.join(OUTPUT_DIR, "video_frame_metadata.json"),
            os.path.join(OUTPUT_DIR, "extraction_manifest.json")
        ]
        
        # Debug info: show which files exist
        existing_files = [f for f in index_files if os.path.exists(f)]
        missing_files = [f for f in index_files if not os.path.exists(f)]
        
        self.existing_index_detected = all(os.path.exists(f) for f in index_files)
        
        if self.existing_index_detected:
            self.log_message_to_queue(f"📋 Existing index detected in {OUTPUT_DIR}")
            self.log_message_to_queue(f"    ✅ Found all required index files:")
            for f in existing_files:
                self.log_message_to_queue(f"        • {os.path.basename(f)}")
            self._setup_existing_index_ui()
        else:
            self.log_message_to_queue(f"📋 No complete index found - ready for new indexing")
            if existing_files:
                self.log_message_to_queue(f"    ⚠️  Some index files exist but incomplete:")
                for f in existing_files:
                    self.log_message_to_queue(f"        • {os.path.basename(f)}")
            if missing_files:
                self.log_message_to_queue(f"    ❌ Missing required files:")
                for f in missing_files:
                    self.log_message_to_queue(f"        • {os.path.basename(f)}")
            self._setup_new_index_ui()

    def _setup_existing_index_ui(self):
        """Configure UI for when an existing index is detected"""
        # Update main buttons with maximum contrast visibility - Labels don't have state/command
        print("DEBUG: _setup_existing_index_ui called - configuring LABELS to BLACK and GREEN")
        self.select_dir_button.config(text="📁 Change Video Directory", bg="#000000", fg="#FFFFFF", font=("Arial", 10, "bold"))
        self.start_button.config(text="🚀 Start Server with Existing Index", bg="#00AA00", fg="#FFFFFF", font=("Arial", 10, "bold"))
        # Bind the start button to start server only
        self.start_button.unbind("<Button-1>")
        self.start_button.bind("<Button-1>", lambda e: self._start_server_only())
        print("DEBUG: Label configuration complete - ready for manual server start")
        
        # Show management tools immediately
        self.management_frame.pack(pady=10, padx=10, fill=tk.X, before=self.dir_label)
        
        # Update directory label to show index location
        self.dir_label_var.set(f"Index ready in: {os.path.abspath(OUTPUT_DIR)}")
        
        # Add information about existing index - but don't auto-start
        self.log_message_to_queue("🎯 Existing index found! Ready to start server.")
        self.log_message_to_queue("   ℹ️  Click 'Start Server' to begin or 'Clear All Indexes' to start fresh")
        self.log_message_to_queue("   ℹ️  Use 'Change Video Directory' to re-index different videos")

    def _do_nothing(self):
        """Empty method for buttons that should appear inactive but not be disabled"""
        pass

    def _setup_new_index_ui(self):
        """Configure UI for when no existing index is found"""
        # Ensure proper button styling for new index mode with high contrast
        self.select_dir_button.config(text="1. Select Video Directory", bg="#000000", fg="#FFFFFF", font=("Arial", 10, "bold"))
        self.start_button.config(text="2. Start Processing & Server", bg="#0000FF", fg="#FFFFFF", font=("Arial", 10, "bold"))

    def _start_server_only(self):
        """Start only the server using existing index"""
        if not self.existing_index_detected:
            messagebox.showerror("Error", "No existing index found. Please run full processing first.")
            return
        
        # Labels don't have state or command - use visual feedback and rebind
        self.start_button.config(bg="#666666", fg="#AAAAAA")  # Visual "disabled" appearance
        self.start_button.unbind("<Button-1>")
        self.start_button.bind("<Button-1>", lambda e: self._do_nothing())
        self._show_management_tools_temporarily()
        
        # Clear previous output
        self.output_area.configure(state='normal')
        self.output_area.delete(1.0, tk.END)
        self.output_area.configure(state='disabled')
        self.log_message_to_queue("--- Starting server with existing index ---")

        # Create a new thread to start just the server
        thread = threading.Thread(target=self._start_server_sequence)
        thread.daemon = True
        thread.start()

    def _start_server_sequence(self):
        """Start just the server without processing"""
        self._terminate_existing_server()
        
        # Kill any other servers that might be running
        self.log_message_to_queue("🔍 Checking for other video search servers...")
        self._kill_servers_silent()

        # Start the FastAPI server
        self.log_message_to_queue("--- Starting: Video Search Server ---")
        server_command_args = [self.python_executable, "-u", SERVER_SCRIPT]
        if self.use_fp16.get():
            server_command_args.append("--fp16")
            self.log_message_to_queue("Using FP16 precision for search server (reduces memory usage)")
        self.log_message_to_queue(f"Executing: {' '.join(server_command_args)}")
        try:
            # Set environment variable for index suffix
            env_vars = os.environ.copy()
            env_vars["VIDEO_INDEXER_SUFFIX"] = ""
            self.log_message_to_queue(f"Set VIDEO_INDEXER_SUFFIX='{env_vars['VIDEO_INDEXER_SUFFIX']}' for server process.")

            creation_flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            pipe_server_output = True if os.name == 'nt' else False

            self.server_process = subprocess.Popen(
                server_command_args, 
                cwd=self.script_dir, 
                creationflags=creation_flags, 
                stdout=subprocess.PIPE if pipe_server_output else None, 
                stderr=subprocess.PIPE if pipe_server_output else None,
                text=True if pipe_server_output else False,
                env=env_vars
            )
            self.log_message_to_queue(f"Server process initiated (PID: {self.server_process.pid}).")

            # Brief pause to see if server crashes immediately
            time.sleep(2.0)

            if self.server_process.poll() is not None:
                self.log_message_to_queue("ERROR: Server process terminated unexpectedly soon after start.")
                if pipe_server_output:
                    stdout, stderr = self.server_process.communicate()
                    self.log_message_to_queue(f"Server STDOUT:\n{stdout}")
                    self.log_message_to_queue(f"Server STDERR:\n{stderr}")
                    if "Address already in use" in stderr or "Errno 48" in stderr:
                        self.log_message_to_queue("Details: Port 8002 is likely already in use.")
                        messagebox.showerror("Server Startup Error", "Port 8002 is already in use. Close other conflicting applications and try again.")
                    else:
                        messagebox.showerror("Server Startup Error", "Server failed to start. Check terminal for errors if not piped, or launcher log if piped.")
                else:
                    messagebox.showerror("Server Startup Error", "Server failed to start. Check the terminal from which this launcher was started for server error messages.")
                
                self.root.after(0, self.reset_buttons_after_error)
                self.server_process = None
                self._update_server_status(False)
                return
            else:
                if pipe_server_output:
                    self.log_message_to_queue("Server process (piped output) appears to be running.")
                else:
                    self.log_message_to_queue("Server process (direct output) appears to be running. Its logs should appear in this terminal.")
                
                # Update status to show server is running
                self.root.after(0, lambda: self._update_server_status(True, SERVER_URL))

        except Exception as e:
            self.log_message_to_queue(f"Error starting server: {e}")
            self.root.after(0, lambda: messagebox.showerror("Server Error", f"Failed to start the web server: {e}"))
            self.root.after(0, self.reset_buttons_after_error)
            self.server_process = None
            self._update_server_status(False)
            return
            
        # Wait for server to be ready and then open browser
        self.log_message_to_queue(f"Waiting for the server to be ready before opening browser...")
        if self._wait_for_server_ready(SERVER_URL, timeout=30):
            try:
                webbrowser.open(SERVER_URL)
                self.log_message_to_queue(f"✅ Server ready! Opened {SERVER_URL} in your web browser.")
            except Exception as e:
                self.log_message_to_queue(f"Could not open web browser: {e}. Please manually navigate to {SERVER_URL}")
        else:
            self.log_message_to_queue(f"⚠️  Server did not become ready in time. Please manually navigate to {SERVER_URL}")
            self.log_message_to_queue("   The server may still be starting up in the background.")

        self.log_message_to_queue("--- Server started with existing index! Video search is now available. ---")
        self.log_message_to_queue(f"🌐 Video search interface: {SERVER_URL}")
        self.log_message_to_queue("⚠️  IMPORTANT: Closing this launcher will stop the video search server!")
        
        # Re-enable buttons (Labels)
        def update_restart_button():
            self.start_button.config(text="🔄 Restart Server", bg="#00AA00", fg="#FFFFFF", font=("Arial", 10, "bold"))
            self.start_button.unbind("<Button-1>")
            self.start_button.bind("<Button-1>", lambda e: self._start_server_only())
        self.root.after(0, update_restart_button)

    def _wait_for_server_ready(self, url, timeout=30):
        """Wait for the server to be ready to accept connections"""
        import urllib.request
        import urllib.error
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to make a simple request to the server
                response = urllib.request.urlopen(url, timeout=2)
                if response.getcode() == 200:
                    return True
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                # Server not ready yet, wait a bit
                time.sleep(0.5)
                continue
            except Exception:
                # Other errors, keep trying
                time.sleep(0.5)
                continue
                
        return False

    def kill_old_servers(self):
        """Kill any existing video indexer processes including servers and multiprocessing"""
        self.log_message_to_queue("🔍 Searching for video indexer processes...")
        
        try:
            import psutil
            killed_count = 0
            
            # Patterns to look for in process command lines
            patterns = [
                'video_search_server',
                'frame_extractor.py',
                'frame_indexer.py', 
                'transcript_extractor',
                'transcript_indexer.py',
                'multiprocessing.resource_tracker',
                'multiprocessing.spawn'
            ]
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline:
                        cmdline_str = ' '.join(str(arg) for arg in cmdline)
                        
                        # Check if this process matches any of our patterns
                        for pattern in patterns:
                            if pattern in cmdline_str and self.script_dir in cmdline_str:
                                self.log_message_to_queue(f"🔪 Killing process PID {proc.info['pid']}: {pattern}")
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except psutil.TimeoutExpired:
                                    proc.kill()
                                killed_count += 1
                                break
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Additional fallback cleanup
            self._kill_video_indexer_processes()
            
            if killed_count > 0:
                self.log_message_to_queue(f"✅ Killed {killed_count} video indexer process(es)")
                messagebox.showinfo("Success", f"Killed {killed_count} video indexer process(es).")
            else:
                self.log_message_to_queue("ℹ️  No old video indexer processes found")
                messagebox.showinfo("Info", "No old video indexer processes found running.")
                
        except ImportError:
            # Fallback method using ps and kill commands
            self.log_message_to_queue("📦 psutil not available, using system commands...")
            self._kill_servers_fallback()
        except Exception as e:
            self.log_message_to_queue(f"❌ Error searching for processes: {e}")
            messagebox.showerror("Error", f"Error searching for old processes: {e}")

    def _kill_servers_fallback(self):
        """Fallback method to kill servers using system commands"""
        try:
            import subprocess
            # Find processes containing video_search_server
            result = subprocess.run(['pgrep', '-f', 'video_search_server'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                killed_count = 0
                
                for pid in pids:
                    if pid.strip():
                        self.log_message_to_queue(f"🔪 Killing server process PID {pid}")
                        subprocess.run(['kill', pid], check=False)
                        killed_count += 1
                
                if killed_count > 0:
                    self.log_message_to_queue(f"✅ Killed {killed_count} old server process(es)")
                    messagebox.showinfo("Success", f"Killed {killed_count} old video search server(s).")
                else:
                    messagebox.showinfo("Info", "No old video search servers found running.")
            else:
                self.log_message_to_queue("ℹ️  No old video search servers found")
                messagebox.showinfo("Info", "No old video search servers found running.")
                
        except Exception as e:
            self.log_message_to_queue(f"❌ Error with fallback method: {e}")
            messagebox.showerror("Error", f"Error killing old servers: {e}")

    def _kill_servers_silent(self):
        """Kill servers silently without showing dialogs (for use during processing)"""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'video_search_server'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                killed_count = 0
                
                for pid in pids:
                    if pid.strip():
                        self.log_message_to_queue(f"🔪 Auto-killing server process PID {pid}")
                        subprocess.run(['kill', pid], check=False)
                        killed_count += 1
                
                if killed_count > 0:
                    self.log_message_to_queue(f"✅ Auto-killed {killed_count} old server process(es)")
                else:
                    self.log_message_to_queue("ℹ️  No old servers found to kill")
            else:
                self.log_message_to_queue("ℹ️  No old video search servers found")
                
        except Exception as e:
            self.log_message_to_queue(f"⚠️  Error auto-killing servers: {e}")

    def stop_current_server(self):
        """Stop the current server started by this launcher"""
        if self.server_process and self.server_process.poll() is None:
            result = messagebox.askyesno("Stop Server", 
                                       "This will stop the video search server.\n"
                                       "You will need to restart processing to use search again.\n\n"
                                       "Stop the server?")
            if result:
                self._terminate_existing_server()
                self._update_server_status(False)
                messagebox.showinfo("Success", "Video search server stopped.")
        else:
            messagebox.showinfo("Info", "No server is currently running from this launcher.")

    def stop_all_processes(self):
        """Stop all active processes (server + processing tasks)"""
        server_running = self.server_process and self.server_process.poll() is None
        processing_active = any(p.poll() is None for p in self.active_processes)
        
        if not server_running and not processing_active:
            messagebox.showinfo("Info", "No active processes to stop.")
            return
        
        processes_info = []
        if server_running:
            processes_info.append("• Video search server")
        if processing_active:
            processes_info.append("• Processing tasks")
        
        message = "This will stop the following processes:\n\n" + "\n".join(processes_info)
        message += "\n\nContinue?"
        
        result = messagebox.askyesno("Stop All Processes", message)
        if result:
            self.log_message_to_queue("🛑 Stopping all processes...")
            terminated_count = self._terminate_all_processes()
            self._update_server_status(False)
            self._update_button_states()
            messagebox.showinfo("Success", f"Stopped {terminated_count} process(es).")

    def _update_button_states(self):
        """Update button states based on active processes"""
        server_running = self.server_process and self.server_process.poll() is None
        processing_active = any(p.poll() is None for p in self.active_processes)
        
        # Stop Server button
        self.stop_server_button.config(state=tk.NORMAL if server_running else tk.DISABLED)
        
        # Stop All button  
        self.stop_all_button.config(state=tk.NORMAL if (server_running or processing_active) else tk.DISABLED)

    def _update_server_status(self, is_running, url=None):
        """Update the server status indicator"""
        if is_running:
            status_text = f"🟢 Server: Running"
            if url:
                status_text += f" at {url}"
            self.server_status_label.config(text=status_text, fg="#4CAF50")
            
            # Show server status and management tools when server is running
            self.server_status_frame.pack(pady=10, padx=10, fill=tk.X, before=self.dir_label)
            self.management_frame.pack(pady=10, padx=10, fill=tk.X, before=self.dir_label)
        else:
            self.server_status_label.config(text="🔴 Server: Not Running", fg="#f44336")
            
            # Hide server status and management tools when no server
            self.server_status_frame.pack_forget()
            self.management_frame.pack_forget()
        
        # Update button states
        self._update_button_states()

    def _show_management_tools_temporarily(self):
        """Show management tools during processing (even if server isn't running yet)"""
        if not self.management_frame.winfo_viewable():
            self.management_frame.pack(pady=10, padx=10, fill=tk.X, before=self.dir_label)

    def _hide_management_if_no_server(self):
        """Hide management tools if no server is running and no processing is active"""
        server_running = self.server_process and self.server_process.poll() is None
        processing_active = any(p.poll() is None for p in self.active_processes)
        
        if not server_running and not processing_active:
            self.management_frame.pack_forget()

    def _terminate_all_processes(self):
        """Terminate all active processes including server and processing tasks"""
        processes_terminated = 0
        
        # Terminate server process
        if self.server_process and self.server_process.poll() is None:
            self.log_message_to_queue(f"🛑 Stopping server process (PID: {self.server_process.pid})...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=3)
                processes_terminated += 1
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                processes_terminated += 1
            except Exception as e:
                self.log_message_to_queue(f"Error terminating server: {e}")
            finally:
                self.server_process = None
        
        # Terminate all active processing tasks
        for process in self.active_processes[:]:  # Copy list to avoid modification during iteration
            if process.poll() is None:  # Process is still running
                self.log_message_to_queue(f"🛑 Stopping processing task (PID: {process.pid})...")
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                        processes_terminated += 1
                    except subprocess.TimeoutExpired:
                        self.log_message_to_queue(f"Process {process.pid} didn't terminate, forcing kill...")
                        process.kill()
                        process.wait(timeout=2)
                        processes_terminated += 1
                except Exception as e:
                    self.log_message_to_queue(f"Error terminating process {process.pid}: {e}")
            
            # Remove from active list regardless
            try:
                self.active_processes.remove(process)
            except ValueError:
                pass  # Already removed
        
        if processes_terminated > 0:
            self.log_message_to_queue(f"✅ Terminated {processes_terminated} process(es)")
        
        # Use system kill as fallback for any remaining processes
        self._kill_video_indexer_processes()
        
        return processes_terminated

    def _kill_video_indexer_processes(self):
        """Kill any remaining video indexer processes using system commands"""
        try:
            # Find processes related to video indexer scripts and multiprocessing
            patterns_to_kill = [
                "frame_extractor.py",
                "frame_indexer.py", 
                "transcript_extractor_pywhisper.py",
                "transcript_indexer.py",
                "video_search_server_transcode.py",
                "multiprocessing.resource_tracker",
                "multiprocessing.spawn"
            ]
            
            killed_count = 0
            for pattern in patterns_to_kill:
                try:
                    # Use pgrep to find processes
                    result = subprocess.run(['pgrep', '-f', pattern], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            if pid.strip():
                                try:
                                    # First try gentle termination
                                    subprocess.run(['kill', '-TERM', pid.strip()], timeout=2)
                                    time.sleep(0.5)
                                    
                                    # Check if still running, then force kill
                                    check_result = subprocess.run(['kill', '-0', pid.strip()], 
                                                               capture_output=True, timeout=1)
                                    if check_result.returncode == 0:  # Process still exists
                                        subprocess.run(['kill', '-9', pid.strip()], timeout=2)
                                        self.log_message_to_queue(f"🔪 Force killed {pattern} process (PID: {pid})")
                                    else:
                                        self.log_message_to_queue(f"✅ Terminated {pattern} process (PID: {pid})")
                                    killed_count += 1
                                except Exception:
                                    # Process might have already died
                                    pass
                except Exception as e:
                    # Silently continue if pgrep/kill fails
                    pass
            
            # Also kill any processes running from this video-indexer directory
            try:
                cmd = ['pgrep', '-f', self.script_dir]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid.strip() and pid.strip() != str(os.getpid()):  # Don't kill ourselves
                            try:
                                subprocess.run(['kill', '-TERM', pid.strip()], timeout=2)
                                time.sleep(0.5)
                                subprocess.run(['kill', '-9', pid.strip()], timeout=2)
                                self.log_message_to_queue(f"🔪 Cleaned up video-indexer process (PID: {pid})")
                                killed_count += 1
                            except Exception:
                                pass
            except Exception:
                pass
                
            if killed_count > 0:
                self.log_message_to_queue(f"🧹 Total processes cleaned up: {killed_count}")
                
        except Exception as e:
            # Silently continue if system commands fail
            pass

    def run_command(self, command, description, callback=None):
        """Runs a command synchronously and logs its output. Should be called from within a thread."""
        self.root.after(0, lambda: self.progress_bar.configure(value=0))
        self.root.after(0, lambda: self.progress_label.config(text=f"{description}: Starting..."))
        self.current_operation = description
        
        self.log_message_to_queue(f"--- Starting: {description} ---")
        self.log_message_to_queue(f"Executing: {' '.join(command)}")
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1, universal_newlines=True, cwd=self.script_dir)
            
            # Track this process for cleanup
            self.active_processes.append(process)
            
            for line in process.stdout:
                self.log_message_to_queue(line.strip())
            process.wait()
            
            # Remove from active processes when complete
            if process in self.active_processes:
                self.active_processes.remove(process)
            if process.returncode == 0:
                self.root.after(0, lambda: self.progress_bar.configure(value=100))
                self.root.after(0, lambda: self.progress_label.config(text=f"{description}: Complete"))
                self.log_message_to_queue(f"--- Finished: {description} (Success) ---")
                if callback: self.root.after(0, lambda: callback(True))
                return True
            else:
                self.root.after(0, lambda: self.progress_label.config(text=f"{description}: Failed"))
                self.log_message_to_queue(f"--- Error: {description} (Failed with code {process.returncode}) ---")
                if callback: self.root.after(0, lambda: callback(False))
                return False
        except FileNotFoundError:
            self.log_message_to_queue(f"Error: Command '{command[0]}' not found for '{description}'.")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Command '{command[0]}' not found. Please ensure Python and necessary scripts are in the correct location."))
            if callback: self.root.after(0, lambda: callback(False))
            return False
        except Exception as e:
            self.log_message_to_queue(f"An unexpected error occurred during '{description}': {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {e}"))
            if callback: self.root.after(0, lambda: callback(False))
            return False

    def start_full_process(self):
        if not self.video_directory:
            messagebox.showerror("Error", "Please select a video directory first.")
            return
        # Disable buttons during full processing (Labels use visual feedback)
        self.select_dir_button.set_enabled(False)
        self.start_button.set_enabled(False)
        
        # Show management tools during processing (even without server)
        self._show_management_tools_temporarily()
        
        # Clear previous output
        self.output_area.configure(state='normal')
        self.output_area.delete(1.0, tk.END)
        self.output_area.configure(state='disabled')
        self.log_message_to_queue("--- Starting new processing sequence ---")

        # Create a new thread to run the sequence of commands
        thread = threading.Thread(target=self._processing_sequence)
        thread.daemon = True # Allow main window to exit even if thread is running
        thread.start()

    def _terminate_existing_server(self):
        if self.server_process and self.server_process.poll() is None: # Check if process exists and is running
            self.log_message_to_queue(f"Attempting to terminate existing server process (PID: {self.server_process.pid})...")
            try:
                self.server_process.terminate() # Send SIGTERM
                self.server_process.wait(timeout=5) # Wait for it to terminate
                self.log_message_to_queue("Existing server process terminated.")
            except subprocess.TimeoutExpired:
                self.log_message_to_queue("Server process did not terminate in time, trying to kill...")
                self.server_process.kill() # Send SIGKILL
                self.server_process.wait(timeout=5)
                self.log_message_to_queue("Server process killed.")
            except Exception as e:
                self.log_message_to_queue(f"Error terminating existing server process: {e}")
            finally:
                self.server_process = None
                self._update_server_status(False)
        elif self.server_process:
             self.log_message_to_queue(f"Previous server process (PID: {self.server_process.pid}) already terminated.")
             self.server_process = None
             self._update_server_status(False)

    def _processing_sequence(self):
        self._terminate_existing_server() # Terminate any server this launcher started
        
        # Also kill any other video search servers that might be running
        self.log_message_to_queue("🔍 Checking for other video search servers...")
        self._kill_servers_silent()  # Silent version for automatic processing

        # 1. Install dependencies
        if os.path.exists(REQUIREMENTS_FILE):
            pip_command = [self.python_executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
            if not self.run_command(pip_command, "Installing dependencies"):
                self.root.after(0, self.reset_buttons_after_error)
                return
        else:
            self.log_message_to_queue(f"Warning: '{REQUIREMENTS_FILE}' not found. Skipping dependency installation.")

        # Ensure output directory exists (though scripts should also handle this)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.log_message_to_queue(f"Ensured output directory exists: {OUTPUT_DIR}")

        # 2. Run frame_extractor.py
        extract_command = [self.python_executable, FRAME_EXTRACTOR_SCRIPT, self.video_directory, "--output-dir", OUTPUT_DIR]
        if not self.run_command(extract_command, "Extracting frames"):
            self.root.after(0, self.reset_buttons_after_error)
            return

        # 3. Run frame_indexer.py
        index_command = [self.python_executable, FRAME_INDEXER_SCRIPT, "--input-dir", OUTPUT_DIR]
        if self.use_fp16.get():
            index_command.append("--fp16")
            self.log_message_to_queue("Using FP16 precision for frame indexing (reduces memory usage)")
        if not self.run_command(index_command, "Indexing frames"):
            self.root.after(0, self.reset_buttons_after_error)
            return

        # 4. Run transcript_extractor_pywhisper.py (with GPU acceleration)
        transcript_extract_command = [self.python_executable, "transcript_extractor_pywhisper.py", self.video_directory, "--output-dir", OUTPUT_DIR, "--whisper-model", "large-v3", "--workers", "1"]
        if not self.run_command(transcript_extract_command, "Extracting transcripts with GPU acceleration"):
            self.log_message_to_queue("Warning: Transcript extraction failed. Continuing without transcripts...")
            # Don't return here - continue without transcripts

        # 5. Run transcript_indexer.py (only if transcript extraction succeeded)
        transcript_manifest_path = os.path.join(OUTPUT_DIR, "transcript_manifest.json")
        if os.path.exists(transcript_manifest_path):
            transcript_index_command = [self.python_executable, "transcript_indexer.py", "--input-dir", OUTPUT_DIR]
            if not self.run_command(transcript_index_command, "Indexing transcripts"):
                self.log_message_to_queue("Warning: Transcript indexing failed. Continuing without transcript search...")
                # Don't return here - continue without transcript search
        else:
            self.log_message_to_queue("No transcript manifest found, skipping transcript indexing.")

        # 6. Start the FastAPI server
        # The server will run in its own process. We won't wait for it here in the same way.
        self.log_message_to_queue("--- Starting: Video Search Server ---")
        server_command_args = [self.python_executable, "-u", SERVER_SCRIPT]
        if self.use_fp16.get():
            server_command_args.append("--fp16")
            self.log_message_to_queue("Using FP16 precision for search server (reduces memory usage)")
        self.log_message_to_queue(f"Executing: {' '.join(server_command_args)}")
        try:
            # Set environment variable for index suffix
            env_vars = os.environ.copy()
            env_vars["VIDEO_INDEXER_SUFFIX"] = ""
            self.log_message_to_queue(f"Set VIDEO_INDEXER_SUFFIX='{env_vars['VIDEO_INDEXER_SUFFIX']}' for server process.")

            creation_flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            
            pipe_server_output = True if os.name == 'nt' else False

            self.server_process = subprocess.Popen(
                server_command_args, 
                cwd=self.script_dir, 
                creationflags=creation_flags, 
                stdout=subprocess.PIPE if pipe_server_output else None, 
                stderr=subprocess.PIPE if pipe_server_output else None,
                text=True if pipe_server_output else False,
                env=env_vars # Pass modified environment
            )
            self.log_message_to_queue(f"Server process initiated (PID: {self.server_process.pid}).")

            # Brief pause to see if server crashes immediately
            time.sleep(2.0) # Increased slightly 

            if self.server_process.poll() is not None: # Process has terminated
                self.log_message_to_queue("ERROR: Server process terminated unexpectedly soon after start.")
                if pipe_server_output:
                    stdout, stderr = self.server_process.communicate()
                    self.log_message_to_queue(f"Server STDOUT:\n{stdout}")
                    self.log_message_to_queue(f"Server STDERR:\n{stderr}")
                    if "Address already in use" in stderr or "Errno 48" in stderr:
                        self.log_message_to_queue("Details: Port 8002 is likely already in use.")
                        messagebox.showerror("Server Startup Error", "Port 8002 is already in use. Close other conflicting applications and try again.")
                    else:
                        messagebox.showerror("Server Startup Error", "Server failed to start. Check terminal for errors if not piped, or launcher log if piped.")
                else: # Not piped, errors should be in the main terminal
                    messagebox.showerror("Server Startup Error", "Server failed to start. Check the terminal from which this launcher was started for server error messages.")
                
                self.root.after(0, self.reset_buttons_after_error)
                self.server_process = None # Clear since it failed
                self._update_server_status(False)
                return
            else:
                if pipe_server_output:
                    self.log_message_to_queue("Server process (piped output) appears to be running.")
                else:
                    self.log_message_to_queue("Server process (direct output) appears to be running. Its logs should appear in this terminal.")
                
                # Update status to show server is running
                self.root.after(0, lambda: self._update_server_status(True, SERVER_URL))

        except Exception as e:
            self.log_message_to_queue(f"Error starting server: {e}")
            self.root.after(0, lambda: messagebox.showerror("Server Error", f"Failed to start the web server: {e}"))
            self.root.after(0, self.reset_buttons_after_error)
            self.server_process = None # Clear on error
            self._update_server_status(False)
            return
            
        # 7. Wait for server to be ready and then open browser
        self.log_message_to_queue(f"Waiting for the server to be ready before opening browser...")
        if self._wait_for_server_ready(SERVER_URL, timeout=30):
            try:
                webbrowser.open(SERVER_URL)
                self.log_message_to_queue(f"✅ Server ready! Opened {SERVER_URL} in your web browser.")
            except Exception as e:
                self.log_message_to_queue(f"Could not open web browser: {e}. Please manually navigate to {SERVER_URL}")
        else:
            self.log_message_to_queue(f"⚠️  Server did not become ready in time. Please manually navigate to {SERVER_URL}")
            self.log_message_to_queue("   The server may still be starting up in the background.")

        self.log_message_to_queue("--- All processes complete! Video search is now available. ---")
        self.log_message_to_queue(f"🌐 Video search interface: {SERVER_URL}")
        self.log_message_to_queue("⚠️  IMPORTANT: Closing this launcher will stop the video search server!")
        # Optionally re-enable buttons or change their text (Labels)
        def restore_buttons():
            self.select_dir_button.set_enabled(True)
            self.start_button.set_enabled(True)
            self.select_dir_button.config(bg="#000000", fg="#FFFFFF")
            self.start_button.config(bg="#0000FF", fg="#FFFFFF")
        self.root.after(0, restore_buttons)
        
        # Hide management tools if server isn't running (processing complete but no server)
        self.root.after(0, lambda: self._hide_management_if_no_server())
        # Or, provide a quit button:
        # self.quit_button = tk.Button(self.root, text="Quit Launcher", command=self.root.quit)
        # self.quit_button.pack(pady=10)


    def _apply_high_contrast_button_colors(self):
        """Apply high contrast colors to ensure buttons are always readable"""
        # Get current button text to determine appropriate colors
        current_select_text = self.select_dir_button.cget("text")
        current_start_text = self.start_button.cget("text")
        
        # Always use high contrast colors
        self.select_dir_button.config(bg="#000000", fg="#FFFFFF", font=("Arial", 10, "bold"))
        
        if "Auto-Starting" in current_start_text:
            self.start_button.config(bg="#FF0000", fg="#FFFFFF", font=("Arial", 10, "bold"))
        else:
            self.start_button.config(bg="#0000FF", fg="#FFFFFF", font=("Arial", 10, "bold"))

    def reset_buttons_after_error(self):
        # Restore Labels to normal appearance and functionality
        self.select_dir_button.set_enabled(True)
        self.start_button.set_enabled(True)
        self.select_dir_button.config(bg="#000000", fg="#FFFFFF")
        self.start_button.config(bg="#0000FF", fg="#FFFFFF")
        self._update_button_states()  # Update stop buttons based on actual process states
        self._hide_management_if_no_server()  # Hide management tools if nothing is running
        messagebox.showinfo("Process Halted", "A step in the process failed. Please check the logs above for details.")

    def on_closing(self):
        """Handle application closing - clean up all processes"""
        server_running = self.server_process and self.server_process.poll() is None
        processing_active = any(p.poll() is None for p in self.active_processes)
        
        if server_running or processing_active:
            processes_info = []
            if server_running:
                processes_info.append("• Video search server")
            if processing_active:
                processes_info.append("• Processing tasks (frame extraction, indexing, etc.)")
            
            message = "The following processes are currently running:\n\n" + "\n".join(processes_info)
            message += "\n\nClosing this launcher will stop all these processes."
            
            if server_running:
                message += "\nThis means video search will no longer be available."
            
            message += "\n\nDo you want to quit and stop all processes?"
            
            result = messagebox.askyesno("Quit Application", message)
            
            if not result:
                return  # User cancelled, don't close
        
        # Always run comprehensive cleanup (even if no tracked processes)
        self.log_message_to_queue("🛑 Shutting down all video indexer processes...")
        try:
            # First terminate tracked processes
            terminated_count = self._terminate_all_processes()
            
            # Then do comprehensive system-wide cleanup
            self._kill_video_indexer_processes()
            
            self.log_message_to_queue(f"✅ Comprehensive cleanup complete. Closing launcher.")
        except Exception as e:
            self.log_message_to_queue(f"⚠️ Error during cleanup: {e}")
        
        # Close the application
        self.root.destroy()


if __name__ == "__main__":
    print("DEBUG: Starting Tkinter application...")
    root = tk.Tk()
    print("DEBUG: tk.Tk() created.")
    app = None
    try:
        app = AppLauncher(root)
        print("DEBUG: AppLauncher instance created successfully.")
        # The bring_to_front_mac call is now scheduled via root.after in AppLauncher.__init__
    except Exception as e:
        print(f"ERROR: Exception during AppLauncher instantiation: {e}")
        import traceback
        traceback.print_exc()
        if root: # Attempt to destroy root if it exists
            try:
                root.destroy()
            except tk.TclError:
                pass # If root is already destroyed or in bad state
        sys.exit(1)
    
    print("DEBUG: Calling root.mainloop()...")
    try:
        if app: # Only run mainloop if app was successfully initialized
            root.mainloop()
        else:
            print("ERROR: App instance is None, not calling mainloop.")
            if root: 
                try: root.destroy() 
                except tk.TclError: pass
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Exception during root.mainloop(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # This line will print when mainloop finishes or if an error above causes exit
        print("DEBUG: root.mainloop() has finished or an error occurred before/during it.") 

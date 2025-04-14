"""
GUI tool for viewing and deleting images in a directory.

Usage:
    python image-viewer-delete.py <directory>
    python image-viewer-delete.py --help
    python image-viewer-delete.py -h

Keyboard shortcuts:
    d: next image
    a: previous image
    w: delete image and land on the next image
    q: quit
    h: help
"""

import os
import sys
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from typing import List, Optional


def get_image_files(directory: str) -> List[Path]:
    """Get a list of image files in the given directory."""
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    return [
        Path(directory) / f
        for f in os.listdir(directory)
        if Path(f).suffix.lower() in image_extensions
    ]


def delete_image(image_path: Path) -> None:
    """Delete the given image file."""
    try:
        os.remove(image_path)
    except Exception as e:
        messagebox.showerror("Error", f"Could not delete image: {e}")
    # else:
    #     messagebox.showinfo("Deleted", f"Deleted image: {image_path.name}")
        print(f"Deleted image: {image_path.name}")


def show_image(image_path: Path, label: tk.Label) -> None:
    """Display the given image in the label after resizing it to match the widget."""
    try:
        image = Image.open(image_path)
        # Ensure the widget's dimensions are updated
        label.update_idletasks()
        # Fallback sizes if needed
        width = label.winfo_width() if label.winfo_width() > 1 else 800
        height = label.winfo_height() if label.winfo_height() > 1 else 600
        # Resize the image to fit the widget
        image = image.resize((width, height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo  # keep reference
    except Exception as e:
        messagebox.showerror("Error", f"Could not open image: {e}")
        print(f"Could not open image: {e}")

def on_resize(event: tk.Event) -> None:
    """Handle resize events on the image label."""
    # If there is an image currently displayed, re-show it with new dimensions
    if image_files:
        show_image(image_files[current_index], image_label)


def next_image() -> None:
    """Show the next image in the list."""
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        show_image(image_files[current_index], image_label)
    else:
        messagebox.showinfo("End", "No more images to show.")


def previous_image() -> None:
    """Show the previous image in the list."""
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image(image_files[current_index], image_label)
    else:
        messagebox.showinfo("Start", "No previous images to show.")


def delete_and_next() -> None:
    """Delete the current image and show the next one."""
    global current_index
    if image_files:
        # open prompt for confirmation
        if not messagebox.askyesno("Delete", "Are you sure you want to delete this image?"):
            return
        delete_image(image_files[current_index])
        image_files.pop(current_index)
        if current_index >= len(image_files):
            current_index = len(image_files) - 1
        if image_files:
            show_image(image_files[current_index], image_label)
        else:
            messagebox.showinfo("End", "No more images to show.")
    else:
        messagebox.showinfo("Empty", "No images to delete.")


def quit_app() -> None:
    """Quit the application."""
    root.quit()
    root.destroy()


def show_help() -> None:
    """Show help message."""
    help_message = (
        "Keyboard shortcuts:\n"
        "d: next image\n"
        "a: previous image\n"
        "w: delete image and land on the next image\n"
        "q: quit\n"
        "h: help"
    )
    messagebox.showinfo("Help", help_message)
    print(help_message)


def on_key_press(event: tk.Event) -> None:
    """Handle key press events."""
    if event.char == "d":
        next_image()
    elif event.char == "a":
        previous_image()
    elif event.char == "w":
        delete_and_next()
    elif event.char == "q":
        quit_app()
    elif event.char == "h":
        show_help()


def on_close() -> None:
    """Handle window close event."""
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        quit_app()


def main(directory: str) -> None:
    """Main function to set up the GUI."""
    global root, image_label, current_index, image_files

    # Initialize the main window
    root = tk.Tk()
    root.title("Image Viewer")
    # main window should have resizable dimensions
    root.geometry("800x600")
    
    root.bind("<Key>", on_key_press)
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Create a label to display images and bind resize event
    image_label = tk.Label(root)
    image_label.pack(fill=tk.BOTH, expand=True)
    image_label.bind("<Configure>", on_resize)

    # Get the list of image files
    image_files = get_image_files(directory)
    if not image_files:
        messagebox.showinfo("No Images", "No images found in the directory.")
        root.quit()
        return

    # Set the current index to 0
    current_index = 0

    # Show the first image
    show_image(image_files[current_index], image_label)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ["--help", "-h"]:
        print("Usage: python image-viewer-delete.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)

    main(directory)
# The script is a GUI tool for viewing and deleting images in a directory.
# It uses Tkinter for the GUI and PIL for image handling.
# The script provides keyboard shortcuts for navigation and deletion of images.
# It also includes error handling for file operations and image loading.

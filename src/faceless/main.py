"""Application entry point helpers for Faceless."""

import tkinter as tk

from .modules.module_a import EnhancedFaceRecognitionSystem


def run_app():
    """Launch the Tkinter-based face recognition system."""
    root = tk.Tk()
    app = EnhancedFaceRecognitionSystem(root)
    root.mainloop()
    return app


def main():
    run_app()


if __name__ == "__main__":
    main()


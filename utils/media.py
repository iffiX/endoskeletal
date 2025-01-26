import os
import numpy as np
import moviepy.editor as mpy
from typing import List
from multiprocessing import get_context


def create_video(
    frames: List[np.ndarray],
    path: str,
    filename: str,
    extension: str = ".gif",
    fps: int = 15,
):
    """
    Args:
        frames: A list of numpy arrays of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the video.
        filename: File name.
        extension: File extension.
        fps: frames per second.
    """
    if frames:
        for f in range(len(frames)):
            if np.issubdtype(frames[f].dtype, np.integer):
                frames[f] = frames[f].astype(np.uint8)
            elif np.issubdtype(frames[f].dtype, np.floating):
                frames[f] = (frames[f] * 255).astype(np.uint8)
            if frames[f].ndim == 2:
                # consider as a grey scale image
                frames[f] = np.repeat(frames[f][:, :, np.newaxis], 3, axis=2)

        clip = mpy.ImageSequenceClip(frames, fps=fps)
        if extension.lower() == ".gif":
            clip.write_gif(
                os.path.join(path, filename + extension),
                fps=fps,
                verbose=False,
                logger=None,
            )
        else:
            clip.write_videofile(
                os.path.join(path, filename + extension),
                fps=fps,
                verbose=False,
                logger=None,
            )
        clip.close()


def create_video_subproc(
    frames: List[np.ndarray],
    path: str,
    filename: str,
    extension: str = ".gif",
    fps: int = 15,
    daemon: bool = True,
):
    """
    Create video with a subprocess, since it takes a lot of time for ``moviepy``
    to encode the video file.
    See Also:
         :func:`.create_video`
    Note:
        if ``daemon`` is true, then this function cannot be used in a
        daemonic subprocess.
    Args:
        frames: A list of numpy arrays of shape (H, W, C) or (H, W), and with
            ``dtype`` = any float or any int.
            When a frame is float type, its value range should be [0, 1].
            When a frame is integer type, its value range should be [0, 255].
        path: Directory to save the video.
        filename: File name.
        extension: File extension.
        fps: frames per second.
        daemon: Whether launching the saving process as a daemonic process.
    Returns:
        A wait function, once called, block until creation has finished.
    """

    def wait():
        pass

    if frames:
        p = get_context("spawn").Process(
            target=create_video, args=(frames, path, filename, extension, fps)
        )
        p.daemon = daemon
        p.start()

        def wait():
            p.join()

    return wait

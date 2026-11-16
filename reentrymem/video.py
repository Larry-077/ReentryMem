from pathlib import Path
import numpy as np


def read_condition(path, count=77, width=832, height=480):
    import av
    from PIL import Image

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else None
        for frame in container.decode(stream):
            pixels = frame.to_ndarray(format="rgb24")
            if pixels.shape[:2] != (height, width):
                pixels = np.asarray(
                    Image.fromarray(pixels).resize((width, height), Image.Resampling.BILINEAR)
                )
            frames.append(pixels)
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f"Expected at least {count} observed frames, decoded {len(frames)}")
    if rate is not None and abs(rate - 15) > 0.01:
        raise ValueError(f"Input is {rate:g} fps; resample the condition clip to 15 fps first")
    return np.stack(frames)


def write_video(path, frames, fps=15):
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = (frames.shape[2], frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Processor for ARC-Hunyuan-Video-7B model."""

import math

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

HUNYUAN_MEAN = (0.48145466, 0.4578275, 0.40821073)
HUNYUAN_STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULT_PATCH_SIZE = 16


def sec2hms(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def add_timestamp_to_frame(
    frame: Image.Image, start_sec: float, end_sec: float
) -> Image.Image:
    """Add timestamp overlay to a video frame."""
    draw = ImageDraw.Draw(frame)
    font_size = int(frame.height * 0.05)
    try:
        font = ImageFont.truetype("ARIAL.TTF", font_size)
    except OSError:
        font = ImageFont.load_default()

    text = f"{sec2hms(start_sec)}-{sec2hms(end_sec)}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = frame.width - text_w - 20
    y = 20
    draw.rectangle(
        [x - 10, y - 10, x + text_w + 10, y + text_h + 10],
        fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return frame


def build_video_transform(image_size: int = 640) -> T.Compose:
    """Build the video frame transform pipeline."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=HUNYUAN_MEAN, std=HUNYUAN_STD),
        ]
    )


def patchify_frames(
    pixel_values: torch.Tensor, patch_size: int = DEFAULT_PATCH_SIZE
) -> torch.Tensor:
    """Convert video frames to flattened patches for the vision encoder.

    HunYuanVisionPatchEmbed expects [total_patches, C * patch_size * patch_size].
    This extracts patches in raster-scan order matching the position embeddings.

    Args:
        pixel_values: Tensor of shape [num_frames, C, H, W].
        patch_size: Size of each square patch.

    Returns:
        Tensor of shape [total_patches, C * patch_size * patch_size]
        where total_patches = num_frames * (H // patch_size) * (W // patch_size).
    """
    n, c, h, w = pixel_values.shape
    grid_h, grid_w = h // patch_size, w // patch_size

    # [N, C, grid_h, patch_size, grid_w, patch_size]
    patches = pixel_values.reshape(n, c, grid_h, patch_size, grid_w, patch_size)
    # [N, grid_h, grid_w, C, patch_size, patch_size]
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    # [N * grid_h * grid_w, C * patch_size * patch_size]
    return patches.reshape(-1, c * patch_size * patch_size)


def process_video_frames(
    frames: list[Image.Image],
    video_duration: float,
    video_fps: float,
    max_num_frame: int = 150,
    image_size: int = 640,
) -> torch.Tensor:
    """Process video frames with timestamp overlay and normalization.

    Matches the reference demo's _calculate_frame_indices + load_video_frames logic:
    - For videos ≤ max_num_frame seconds: 1 frame per second
    - For videos > max_num_frame seconds: max_num_frame evenly-spaced frames
    - Timestamp overlay uses actual video timing

    Args:
        frames: List of PIL Image frames (already extracted by video backend).
        video_duration: Actual video duration in seconds (from metadata).
        video_fps: Actual video FPS (from metadata).
        max_num_frame: Maximum number of frames.
        image_size: Target image size.

    Returns:
        Tensor of shape [num_frames, 3, image_size, image_size].
    """
    num_frames = len(frames)
    transform = build_video_transform(image_size)

    # Compute time intervals matching the reference logic exactly:
    # Reference: _calculate_frame_indices uses actual fps and duration
    if video_duration <= max_num_frame:
        # 1 second intervals, ceil(duration) frames
        interval = 1.0
        num_target_frames = min(num_frames, math.ceil(video_duration))
        intervals_sec = [
            (int(i * interval), int((i + 1) * interval))
            for i in range(num_target_frames)
        ]
    else:
        # Evenly-spaced over entire duration, max_num_frame segments
        num_target_frames = min(num_frames, max_num_frame)
        segment_duration = video_duration / num_target_frames
        intervals_sec = [
            (round(i * segment_duration), round((i + 1) * segment_duration))
            for i in range(num_target_frames)
        ]

    # Uniformly subsample frames, taking the middle frame of each interval
    # This matches the reference's _calculate_frame_indices which picks
    # frame_indices.append((start + end) // 2) for each interval.
    if num_frames > num_target_frames:
        segment_size = num_frames / num_target_frames
        selected_indices = [
            int((i * segment_size + (i + 1) * segment_size) // 2)
            for i in range(num_target_frames)
        ]
        frames = [frames[idx] for idx in selected_indices]
    else:
        frames = frames[:num_target_frames]

    pixel_values = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.asarray(frame))
        start_sec, end_sec = intervals_sec[i]
        frame = add_timestamp_to_frame(frame, start_sec, end_sec)
        pixel_values.append(transform(frame))

    return torch.stack(pixel_values).to(torch.bfloat16)


def process_audio(
    audio: np.ndarray,
    sr: int,
    feature_extractor,
    max_num_frame: int = 150,
    max_total_sec: int = 300,
) -> tuple[torch.Tensor, int]:
    """Process audio waveform into mel spectrogram features.

    Matches reference demo's process_audio + cut_audio_with_librosa logic.

    Args:
        audio: Audio waveform as numpy array.
        sr: Sample rate.
        feature_extractor: WhisperFeatureExtractor instance.
        max_num_frame: Maximum number of frames.
        max_total_sec: Maximum audio duration in seconds.

    Returns:
        Tuple of (spectrogram_features, duration_in_seconds).
    """
    # Ensure mono
    if len(audio.shape) == 2:
        audio = audio[:, 0]

    total_sec = len(audio) / sr

    # Cut audio if too long (reference: cut_audio_with_librosa)
    if total_sec > max_total_sec:
        segment_length = len(audio) // max_num_frame
        segment_samples = int(2 * sr)  # 2 seconds per segment
        segments = []
        for i in range(max_num_frame):
            start = i * segment_length
            end = min(start + segment_samples, len(audio))
            segments.append(audio[start:end])
        audio = np.concatenate(segments)

    # Pad if less than 1 second (reference: _pad_audio)
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)

    duration = math.ceil(len(audio) / sr)

    # Extract spectrograms in 30-second chunks (reference: _extract_spectrogram)
    segment_length = sr * 30
    spectrograms = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i : i + segment_length]
        features = feature_extractor(segment, sampling_rate=sr, return_tensors="pt")[
            "input_features"
        ]
        spectrograms.append(features)

    audio_features = torch.cat(spectrograms).to(torch.bfloat16)
    return audio_features, duration


def generate_silent_audio(
    duration_sec: float,
    sr: int = 16000,
    feature_extractor=None,
    max_total_sec: int = 300,
) -> tuple[torch.Tensor, int]:
    """Generate silent audio features when video has no audio track.

    Matches the reference demo's fallback behavior:
        duration = min(math.ceil(video.duration), 300)
        silent_audio = AudioSegment.silent(duration=duration * 1000)

    Args:
        duration_sec: Video duration in seconds.
        sr: Sample rate.
        feature_extractor: WhisperFeatureExtractor instance.
        max_total_sec: Maximum audio duration in seconds.

    Returns:
        Tuple of (spectrogram_features, duration_in_seconds).
    """
    duration = min(math.ceil(duration_sec), max_total_sec)
    # Create silent audio of the correct duration
    silent_audio = np.zeros(duration * sr, dtype=np.float32)

    return process_audio(
        silent_audio,
        sr,
        feature_extractor,
    )


class ARCHunyuanVideoProcessor(ProcessorMixin):
    """Processor for ARC-Hunyuan-Video-7B model.

    Handles video frame preprocessing (timestamp overlay, resize, normalize)
    and audio feature extraction (WhisperFeatureExtractor).
    """

    attributes = ["tokenizer", "feature_extractor"]
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "WhisperFeatureExtractor"

    def __init__(
        self,
        tokenizer=None,
        feature_extractor=None,
        image_size: int = 640,
        max_num_frame: int = 150,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_size = image_size
        self.max_num_frame = max_num_frame

        # Token IDs
        self.image_token_id = 127968
        if tokenizer is not None:
            self.image_token = tokenizer.convert_ids_to_tokens(self.image_token_id)
        else:
            self.image_token = "<image>"

        super().__init__(tokenizer, feature_extractor, **kwargs)

    def __call__(
        self,
        text=None,
        videos=None,
        audios=None,
        video_metadata=None,
        **kwargs,
    ) -> BatchFeature:
        """Process video + audio + text inputs.

        Args:
            text: Text prompt(s).
            videos: Video frames (list of numpy arrays or PIL Images).
            audios: Audio waveform as (numpy_array, sample_rate) tuple, or None.
            video_metadata: Dict with 'duration' and 'fps' from video backend.
        """
        data = {}

        # Process video frames
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]

            all_pixel_values = []
            all_grid_thw = []

            for video in videos:
                # video is a list of PIL Images or numpy arrays
                if isinstance(video, (list, tuple)):
                    frames = [
                        Image.fromarray(f) if isinstance(f, np.ndarray) else f
                        for f in video
                    ]
                elif isinstance(video, np.ndarray):
                    # Shape: (T, H, W, C)
                    frames = [Image.fromarray(video[i]) for i in range(video.shape[0])]
                else:
                    frames = [video]

                # Get actual video duration/fps from metadata
                # (supplied by video backend or estimated)
                if video_metadata is not None:
                    video_duration = video_metadata.get("duration", len(frames))
                    video_fps = video_metadata.get("fps", 1.0)
                else:
                    # Fallback: estimate from frame count (assumes ~1fps extraction)
                    video_duration = float(len(frames))
                    video_fps = 1.0

                pixel_values = process_video_frames(
                    frames,
                    video_duration=video_duration,
                    video_fps=video_fps,
                    max_num_frame=self.max_num_frame,
                    image_size=self.image_size,
                )
                all_pixel_values.append(pixel_values)

                # Grid THW per frame: raw patch grid (ViT input)
                # patch_size=16, so 640/16 = 40
                grid_h = self.image_size // DEFAULT_PATCH_SIZE
                grid_w = self.image_size // DEFAULT_PATCH_SIZE
                for _ in range(pixel_values.shape[0]):
                    all_grid_thw.append([1, grid_h, grid_w])

            # Patchify: [total_frames, 3, H, W] -> [total_patches, C*P*P]
            # HunYuanVisionPatchEmbed expects pre-patchified input
            frames_tensor = torch.cat(all_pixel_values, dim=0)
            data["pixel_values"] = patchify_frames(
                frames_tensor, patch_size=DEFAULT_PATCH_SIZE
            )
            data["image_grid_thw"] = torch.tensor(all_grid_thw, dtype=torch.int64)

            # If no audio provided, generate silent audio matching video duration
            # This matches the reference demo's fallback behavior.
            if audios is None and self.feature_extractor is not None:
                audio_features, audio_duration = generate_silent_audio(
                    duration_sec=video_duration,
                    feature_extractor=self.feature_extractor,
                )
                data["input_features"] = audio_features
                data["audio_duration"] = torch.tensor(audio_duration)

        # Process audio (if explicitly provided)
        if audios is not None and self.feature_extractor is not None:
            if isinstance(audios, tuple):
                audio_array, audio_sr = audios
            elif isinstance(audios, np.ndarray):
                audio_array = audios
                audio_sr = 16000
            else:
                audio_array = np.array(audios, dtype=np.float32)
                audio_sr = 16000

            audio_features, audio_duration = process_audio(
                audio_array,
                audio_sr,
                self.feature_extractor,
                max_num_frame=self.max_num_frame,
            )
            data["input_features"] = audio_features
            data["audio_duration"] = torch.tensor(audio_duration)

        # Duration-based frame trimming (reference: generate_summary trims
        # pixel_values by audio duration BEFORE building the prompt).
        # This ensures the number of <image> tokens matches actual frames.
        if videos is not None and "audio_duration" in data:
            audio_dur = int(data["audio_duration"].item())
            total_frames = all_pixel_values[0].shape[0] if all_pixel_values else 0
            if audio_dur < total_frames:
                all_pixel_values[0] = all_pixel_values[0][:audio_dur]
                all_grid_thw = all_grid_thw[:audio_dur]
            if audio_dur <= self.max_num_frame and all_pixel_values:
                data["audio_duration"] = torch.tensor(all_pixel_values[0].shape[0])

            # Rebuild pixel_values and grid_thw after trimming
            frames_tensor = torch.cat(all_pixel_values, dim=0)
            data["pixel_values"] = patchify_frames(
                frames_tensor, patch_size=DEFAULT_PATCH_SIZE
            )
            data["image_grid_thw"] = torch.tensor(all_grid_thw, dtype=torch.int64)

        # Process text
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text, add_special_tokens=False, **kwargs)
            data.update(text_inputs)

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + [
                    "pixel_values",
                    "image_grid_thw",
                    "input_features",
                    "audio_duration",
                ]
            )
        )

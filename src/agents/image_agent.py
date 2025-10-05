import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

from .base_agent import BaseAgent, FileAnalysis


class ImageAgent(BaseAgent):
    """Agent that generates captions, tags, and embeddings for image files."""

    def __init__(self):
        self._supported_formats = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".gif",
            ".webp",
        ]
        self._caption_model_name = "Salesforce/blip-image-captioning-base"
        self._clip_model_name = "openai/clip-vit-base-patch32"
        self._caption_model: BlipForConditionalGeneration | None = None
        self._caption_processor: BlipProcessor | None = None
        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None

    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats

    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        """Analyze an image file by delegating to a worker thread."""
        return await asyncio.to_thread(self._process_image, file_path, metadata or {})

    def _process_image(self, file_path: str, metadata: Dict) -> FileAnalysis:
        try:
            self._ensure_models_loaded()
            image = self._load_image(file_path)

            caption = self._generate_caption(image)
            tags = self._generate_tags(caption)
            embedding = self._compute_embedding(image)

            merged_metadata = {
                **metadata,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "caption_model": self._caption_model_name,
                "embedding_model": self._clip_model_name,
            }

            return FileAnalysis(
                summary=caption,
                tags=tags,
                embedding=embedding,
                metadata=merged_metadata,
            )

        except Exception as exc:
            logging.error("Error processing image %s: %s", file_path, exc)
            return FileAnalysis(
                summary=f"Error analyzing image: {exc}",
                tags=["error"],
                embedding=[],
                metadata={"error": str(exc)},
            )

    def _ensure_models_loaded(self) -> None:
        if self._caption_model is None or self._caption_processor is None:
            logging.debug("Loading BLIP model: %s", self._caption_model_name)
            self._caption_processor = BlipProcessor.from_pretrained(self._caption_model_name)
            self._caption_model = BlipForConditionalGeneration.from_pretrained(
                self._caption_model_name
            )
            self._caption_model.eval()

        if self._clip_model is None or self._clip_processor is None:
            logging.debug("Loading CLIP model: %s", self._clip_model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_name)
            self._clip_model = CLIPModel.from_pretrained(self._clip_model_name)
            self._clip_model.eval()

    def _load_image(self, file_path: str) -> Image.Image:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        with Image.open(path) as img:
            return img.convert("RGB")

    def _generate_caption(self, image: Image.Image) -> str:
        assert self._caption_model and self._caption_processor
        inputs = self._caption_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self._caption_model.generate(**inputs, max_length=40)
        return self._caption_processor.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_tags(self, caption: str, max_tags: int = 5) -> List[str]:
        words = [token.strip().lower() for token in caption.replace(",", " ").split()]
        # Deduplicate while preserving order
        seen: set[str] = set()
        tags: List[str] = []
        for word in words:
            if word and word not in seen:
                seen.add(word)
                tags.append(word)
            if len(tags) >= max_tags:
                break
        return tags or ["image"]

    def _compute_embedding(self, image: Image.Image) -> List[float]:
        assert self._clip_model and self._clip_processor
        inputs = self._clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self._clip_model.get_image_features(**inputs)
        normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return normalized.squeeze(0).cpu().tolist()

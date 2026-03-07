"""Extract embedded images from PDF pages using PyMuPDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
import structlog

from src.config.settings import PROJECT_ROOT

logger = structlog.get_logger(__name__)

IMAGES_DIR = PROJECT_ROOT / "data" / "images"

# Minimum dimensions to filter out tiny decorative images
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50


@dataclass
class ExtractedImage:
    """Metadata for a single extracted image."""

    page_num: int  # 1-based
    image_index: int  # Index within the page
    file_path: Path
    width: int
    height: int
    colorspace: str
    xref: int  # PyMuPDF internal reference


def extract_images_from_page(
    doc: fitz.Document,
    page_idx: int,
    doc_id: str,
    output_dir: Path,
) -> list[ExtractedImage]:
    """Extract images from a single PDF page.

    Filters out tiny images (likely decorative elements).
    Saves images as PNG files to the output directory.
    """
    page_num = page_idx + 1
    log = logger.bind(page_num=page_num, doc_id=doc_id)
    results: list[ExtractedImage] = []

    page = doc[page_idx]
    image_list = page.get_images(full=True)

    for img_idx, img_info in enumerate(image_list):
        xref = img_info[0]

        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            log.debug("image_extraction_failed", xref=xref, error=str(exc))
            continue

        width = base_image["width"]
        height = base_image["height"]

        # Skip tiny images
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            continue

        colorspace = base_image.get("colorspace", "unknown")
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")

        # Save to disk
        image_filename = f"{doc_id}_p{page_num}_img{img_idx}.{ext}"
        image_path = output_dir / image_filename
        image_path.write_bytes(image_bytes)

        results.append(
            ExtractedImage(
                page_num=page_num,
                image_index=img_idx,
                file_path=image_path,
                width=width,
                height=height,
                colorspace=str(colorspace),
                xref=xref,
            )
        )

    if results:
        log.info("images_extracted", count=len(results))

    return results


def extract_images(
    file_path: Path,
    doc_id: str,
    page_count: int,
    output_dir: Path | None = None,
) -> list[ExtractedImage]:
    """Extract all images from a PDF document.

    Args:
        file_path: Path to the PDF file.
        doc_id: Document ID for naming output files.
        page_count: Total number of pages in the PDF.
        output_dir: Directory to save extracted images.
                    Defaults to data/images/<doc_id>/

    Returns:
        List of all extracted images across all pages.
    """
    log = logger.bind(file_path=str(file_path), doc_id=doc_id)

    if output_dir is None:
        output_dir = IMAGES_DIR / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[ExtractedImage] = []

    doc = fitz.open(str(file_path))
    for page_idx in range(page_count):
        try:
            images = extract_images_from_page(doc, page_idx, doc_id, output_dir)
            all_images.extend(images)
        except Exception as exc:
            log.warning("image_extraction_error", page_num=page_idx + 1, error=str(exc))

    doc.close()
    log.info("image_extraction_complete", total_images=len(all_images))
    return all_images

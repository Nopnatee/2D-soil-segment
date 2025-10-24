"""Tools for sending raw imagery to Roboflow Annotate.

The Roboflow REST API exposes an ``/upload`` endpoint that accepts images and
optionally places them into the annotation queue. This helper wraps the common
workflow:

1. Collect unlabelled images from disk.
2. Upload them (optionally assigning batches, metadata, or an annotator).
3. Optionally attach paired segmentation masks.
4. Wait for uploads to finish processing so they are ready in the web UI.

The implementation follows the public documentation at
https://docs.roboflow.com/api-reference/upload and keeps the interface thin so
that it can be easily scripted or reused inside notebooks.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import requests


DEFAULT_MASK_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg")


class RoboflowAnnotator:
    """Lightweight client for pushing images into Roboflow Annotate."""

    API_ROOT = "https://api.roboflow.com"

    def __init__(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: Optional[int | str] = None,
        *,
        timeout: float = 60,
        poll_interval: float = 2.5,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Roboflow API key must be provided.")

        self.api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = str(version).lstrip("v") if version is not None else None
        self.dataset_slug = f"{workspace}/{project}"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = session or requests.Session()

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def upload_image(
        self,
        image_path: Path | str,
        *,
        split: str = "train",
        batch_name: Optional[str] = None,
        annotate: bool = True,
        metadata: Optional[Dict[str, object]] = None,
        assignee_email: Optional[str] = None,
        wait_for_processing: bool = True,
        mask_path: Optional[Path | str] = None,
    ) -> Dict[str, object]:
        """Upload a single image to Roboflow using the multipart form endpoint.

        Args:
            image_path: Path to the image file on disk.
            split: Optional dataset split recognised by Roboflow (train/valid/test).
            batch_name: Optional batch label; helpful for grouping related uploads.
            annotate: When True, the image is queued inside Roboflow Annotate.
            metadata: Arbitrary flat key/value metadata stored with the upload.
            assignee_email: Roboflow user e-mail to notify for the annotation job.
            wait_for_processing: Poll the API until the upload finishes processing.
            mask_path: Optional path to a segmentation mask paired with the image.

        Returns:
            The JSON payload describing the upload response.
        """
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        mask_file_path = Path(mask_path).expanduser() if mask_path else None
        if mask_file_path and not mask_file_path.is_file():
            raise FileNotFoundError(f"Mask path does not exist: {mask_file_path}")

        url = f"{self.API_ROOT}/dataset/{self.dataset_slug}/upload"
        params = self._build_params(
            split=split,
            batch=batch_name,
            annotate=annotate,
            assignee=assignee_email,
            metadata=metadata,
            file_name=path.name,
        )
        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "application/octet-stream"

        if mask_file_path:
            mask_mime, _ = mimetypes.guess_type(mask_file_path.name)
            mask_mime = mask_mime or "application/octet-stream"
            with path.open("rb") as image_obj, mask_file_path.open("rb") as mask_obj:
                files = {
                    "file": (path.name, image_obj, mime_type),
                    "mask": (mask_file_path.name, mask_obj, mask_mime),
                }
                response = self.session.post(
                    url,
                    params=params,
                    files=files,
                    timeout=self.timeout,
                )
                data = self._handle_response(response)
        else:
            with path.open("rb") as image_obj:
                files = {"file": (path.name, image_obj, mime_type)}
                response = self.session.post(
                    url,
                    params=params,
                    files=files,
                    timeout=self.timeout,
                )
                data = self._handle_response(response)

        if wait_for_processing and (upload_id := data.get("id") or data.get("upload_id")):
            try:
                processed = self._wait_for_upload(upload_id)
            except RuntimeError:
                # The polling endpoint is optional; fall back to immediate response.
                processed = None
            if isinstance(processed, dict):
                data.update(processed)

        return data

    def upload_directory(
        self,
        directory: Path | str,
        *,
        split: str = "train",
        batch_name: Optional[str] = None,
        annotate: bool = True,
        metadata_builder: Optional[Callable[[Path], Dict[str, object]]] = None,
        assignee_email: Optional[str] = None,
        wait_for_processing: bool = True,
        recursive: bool = True,
        mask_directory: Optional[Path | str] = None,
        mask_suffixes: Sequence[str] = DEFAULT_MASK_EXTS,
        allow_missing_masks: bool = False,
    ) -> Dict[Path, Dict[str, object]]:
        """Upload every image contained in ``directory``.

        Args:
            directory: Folder containing candidate images.
            split: Dataset split tag to apply to each upload.
            batch_name: Optional Roboflow batch name.
            annotate: When True, images are sent directly to Roboflow Annotate.
            metadata_builder: Callable returning metadata per image (optional).
            assignee_email: Optional Roboflow annotator e-mail.
            wait_for_processing: Block until every upload is processed.
            recursive: Recurse into subdirectories when True.
            mask_directory: Directory containing mask images keyed by filename.
            mask_suffixes: Acceptable file extensions for mask lookups.
            allow_missing_masks: When False, raise if a mask cannot be found.

        Returns:
            Mapping of image path to upload metadata.
        """
        directory_path = Path(directory).expanduser()
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Upload directory not found: {directory_path}")

        pattern = "**/*" if recursive else "*"
        uploads: Dict[Path, Dict[str, object]] = {}
        mask_root = Path(mask_directory).expanduser() if mask_directory else None

        for image_file in sorted(directory_path.glob(pattern)):
            if image_file.is_dir():
                continue

            mime_type, _ = mimetypes.guess_type(image_file.name)
            if not mime_type or not mime_type.startswith(("image/", "video/")):
                continue  # Skip non-visual artefacts.

            mask_path = None
            if mask_root:
                mask_path = self._match_mask_path(
                    image_file=image_file,
                    image_root=directory_path,
                    mask_root=mask_root,
                    mask_suffixes=mask_suffixes,
                )
                if not mask_path and not allow_missing_masks:
                    raise FileNotFoundError(
                        f"Mask not found for {image_file.name} in {mask_root}"
                    )

            metadata = metadata_builder(image_file) if metadata_builder else None
            uploads[image_file] = self.upload_image(
                image_file,
                split=split,
                batch_name=batch_name,
                annotate=annotate,
                metadata=metadata,
                assignee_email=assignee_email,
                wait_for_processing=wait_for_processing,
                mask_path=mask_path,
            )

        return uploads

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    def _build_params(
        self,
        *,
        split: str,
        batch: Optional[str],
        annotate: bool,
        assignee: Optional[str],
        metadata: Optional[Dict[str, object]],
        file_name: str,
    ) -> Dict[str, str]:
        params: Dict[str, str] = {
            "api_key": self.api_key,
            "split": split,
            "name": file_name,
        }

        if batch:
            params["batch"] = batch
        if annotate:
            params["annotate"] = "true"
        if assignee:
            params["assign"] = assignee
        if metadata:
            params["metadata"] = json.dumps(metadata)
        if self.version:
            params["version"] = self.version
        return params

    def _match_mask_path(
        self,
        *,
        image_path: Path,
        image_root: Path,
        mask_root: Path,
        mask_suffixes: Sequence[str],
    ) -> Optional[Path]:
        """Resolve a mask path for ``image_path`` within ``mask_root``."""
        try:
            relative = image_path.relative_to(image_root)
        except ValueError:
            relative = Path(image_path.name)

        if relative.parent == Path("."):
            search_dirs = [mask_root]
        else:
            search_dirs = [mask_root / relative.parent, mask_root]

        suffix_candidates = []
        if relative.suffix:
            suffix_candidates.append(relative.suffix)
        for suffix in mask_suffixes:
            if suffix not in suffix_candidates:
                suffix_candidates.append(suffix)

        stem = relative.stem
        for base_dir in search_dirs:
            for suffix in suffix_candidates:
                candidate = base_dir / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    def _wait_for_upload(self, upload_id: str) -> Dict[str, object]:
        """Poll Roboflow until the upload finishes processing."""
        status_url = f"{self.API_ROOT}/dataset/{self.dataset_slug}/uploads/{upload_id}"
        params = {"api_key": self.api_key}

        while True:
            response = self.session.get(status_url, params=params, timeout=self.timeout)
            payload = self._handle_response(response)
            status = payload.get("status") or payload.get("state")
            if status in {"pending", "processing", "queued"}:
                time.sleep(self.poll_interval)
                continue
            return payload

    @staticmethod
    def _handle_response(response: requests.Response) -> Dict[str, object]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else "No response body."
            raise RuntimeError(f"Roboflow request failed: {exc}; response={body}") from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Roboflow response did not contain JSON.") from exc


def _env_or_arg(value: Optional[str], env_key: str) -> str:
    return value or os.getenv(env_key) or ""


def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload images to Roboflow Annotate.")
    parser.add_argument("--api-key", help="Roboflow API key (env: ROBOFLOW_API_KEY).")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug.")
    parser.add_argument("--project", required=True, help="Roboflow project slug.")
    parser.add_argument("--version", help="Dataset version to target (e.g. 1).")
    parser.add_argument("--path", required=True, help="File or directory to upload.")
    parser.add_argument("--split", default="train", help="Dataset split tag (default: train).")
    parser.add_argument("--batch", help="Optional batch name for the upload.")
    parser.add_argument(
        "--mask",
        help="Mask file to upload alongside --path when uploading a single image.",
    )
    parser.add_argument(
        "--mask-dir",
        help="Directory containing segmentation masks that mirror the image filenames.",
    )
    parser.add_argument(
        "--mask-ext",
        dest="mask_exts",
        action="append",
        help="Additional mask file extensions to consider (e.g. .tif, tif).",
    )
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="Skip the mask attachment when a matching file cannot be found.",
    )
    parser.add_argument(
        "--assignee",
        help="Roboflow user e-mail to assign for annotation.",
    )
    parser.add_argument(
        "--metadata",
        help="Inline JSON metadata to attach to each upload (only for files).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When uploading a directory, recurse into sub-folders.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_cli_args(argv)
    api_key = _env_or_arg(args.api_key, "ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "API key missing. Provide --api-key or set the ROBOFLOW_API_KEY environment variable."
        )

    annotator = RoboflowAnnotator(
        api_key=api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
    )

    candidate_path = Path(args.path).expanduser()
    metadata = json.loads(args.metadata) if args.metadata else None
    extra_exts = []
    if args.mask_exts:
        for ext in args.mask_exts:
            if not ext:
                continue
            cleaned = ext.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            extra_exts.append(cleaned)
    mask_suffixes = tuple(dict.fromkeys([*DEFAULT_MASK_EXTS, *extra_exts]))

    if candidate_path.is_dir():
        uploads = annotator.upload_directory(
            candidate_path,
            split=args.split,
            batch_name=args.batch,
            assignee_email=args.assignee,
            metadata_builder=(lambda _: metadata) if metadata else None,
            wait_for_processing=True,
            recursive=args.recursive,
            mask_directory=args.mask_dir,
            mask_suffixes=mask_suffixes,
            allow_missing_masks=args.allow_missing_masks,
        )
        for uploaded_path, payload in uploads.items():
            print(f"[{uploaded_path}] -> {payload.get('status', 'done')}")
    else:
        mask_path = Path(args.mask).expanduser() if args.mask else None
        if not mask_path and args.mask_dir:
            mask_root = Path(args.mask_dir).expanduser()
            mask_candidate = annotator._match_mask_path(  # type: ignore[attr-defined]
                image_path=candidate_path,
                image_root=candidate_path.parent,
                mask_root=mask_root,
                mask_suffixes=mask_suffixes,
            )
            mask_path = mask_candidate
            if not mask_path and not args.allow_missing_masks and args.mask_dir:
                raise FileNotFoundError(
                    f"Mask not found for {candidate_path.name} in {mask_root}"
                )

        payload = annotator.upload_image(
            candidate_path,
            split=args.split,
            batch_name=args.batch,
            metadata=metadata,
            assignee_email=args.assignee,
            mask_path=mask_path,
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

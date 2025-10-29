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
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


DEFAULT_MASK_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg")
DEFAULT_ANNOTATION_EXTS: Sequence[str] = (".xml", ".txt", ".json", ".jsonl", ".geojson")


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
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError("Roboflow API key must be provided.")

        self.api_key = api_key

        workspace_slug = workspace.strip().strip("/")
        project_slug = project.strip().strip("/")

        if "/" in project_slug:
            inferred_workspace, inferred_project = project_slug.split("/", 1)
            if workspace_slug and workspace_slug != inferred_workspace:
                raise ValueError(
                    "Workspace mismatch: project parameter encodes a workspace different from --workspace."
                )
            workspace_slug = workspace_slug or inferred_workspace
            project_slug = inferred_project

        if not workspace_slug:
            raise ValueError("Roboflow workspace slug must be provided.")
        if not project_slug:
            raise ValueError("Roboflow project slug must be provided.")

        self.workspace = workspace_slug
        self.project = project_slug
        self.version = str(version).lstrip("v") if version is not None else None
        self.dataset_slug = project_slug
        self.timeout = float(timeout)
        self.poll_interval = poll_interval
        self.session = session or requests.Session()

        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

        timeout_connect = max(5.0, min(self.timeout, 30.0))
        timeout_read = max(5.0, self.timeout)
        self._timeout_tuple = (timeout_connect, timeout_read)

        retry_config = None
        if self.max_retries:
            retry_config = Retry(
                total=self.max_retries,
                read=self.max_retries,
                connect=self.max_retries,
                status=self.max_retries,
                backoff_factor=self.retry_backoff,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "POST"),
                raise_on_status=False,
            )

        adapter = HTTPAdapter(max_retries=retry_config) if retry_config else HTTPAdapter()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

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
        annotation_path: Optional[Path | str] = None,
        annotation_name: Optional[str] = None,
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
            annotation_path: Optional path to an external annotation file (e.g.,
                VOC XML, YOLO TXT, COCO JSON). If provided, this method will
                attach the annotation to the newly uploaded image via the
                `/annotate/{image_id}` endpoint.
            annotation_name: Optional name for the annotation file stored in
                Roboflow. Defaults to the basename of ``annotation_path``.

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
                try:
                    response = self.session.post(
                        url,
                        params=params,
                        files=files,
                        timeout=self._timeout_tuple,
                    )
                except requests.exceptions.Timeout as exc:
                    raise RuntimeError(
                        f"Roboflow upload timed out after {self.timeout:.0f}s (file={path.name}). "
                        "Increase --timeout or reduce batch size and retry."
                    ) from exc
                except requests.exceptions.RequestException as exc:
                    raise RuntimeError(f"Roboflow upload request failed for {path.name}: {exc}") from exc
                data = self._handle_response(response)
        else:
            with path.open("rb") as image_obj:
                files = {"file": (path.name, image_obj, mime_type)}
                try:
                    response = self.session.post(
                        url,
                        params=params,
                        files=files,
                        timeout=self._timeout_tuple,
                    )
                except requests.exceptions.Timeout as exc:
                    raise RuntimeError(
                        f"Roboflow upload timed out after {self.timeout:.0f}s (file={path.name}). "
                        "Use --timeout to increase the limit or retry the upload."
                    ) from exc
                except requests.exceptions.RequestException as exc:
                    raise RuntimeError(f"Roboflow upload request failed for {path.name}: {exc}") from exc
                data = self._handle_response(response)

        upload_id = data.get("id") or data.get("upload_id")
        if wait_for_processing and upload_id:
            try:
                processed = self._wait_for_upload(upload_id)
            except RuntimeError:
                # The polling endpoint is optional; fall back to immediate response.
                processed = None
            if isinstance(processed, dict):
                data.update(processed)

        # Optionally attach an external annotation file using the annotate endpoint.
        if annotation_path:
            # Best-effort resolution of the Roboflow image id.
            image_id: Optional[str] = None
            # Common payload shapes include: {"image": {"id": "..."}} or "image_id" keys.
            image_obj = data.get("image") if isinstance(data, dict) else None
            if isinstance(image_obj, dict) and isinstance(image_obj.get("id"), str):
                image_id = image_obj.get("id")
            elif isinstance(data.get("image_id"), str):  # type: ignore[union-attr]
                image_id = data.get("image_id")  # type: ignore[assignment]
            elif isinstance(data.get("imageId"), str):  # type: ignore[union-attr]
                image_id = data.get("imageId")  # type: ignore[assignment]
            elif isinstance(upload_id, str):
                # Upload responses often return the image id in the "id" field.
                image_id = upload_id

            # If we still don't have an image id but we have an upload id, poll once.
            if not image_id and upload_id:
                try:
                    processed = self._wait_for_upload(str(upload_id))
                    if isinstance(processed, dict):
                        data.update(processed)
                        image_obj = processed.get("image") if isinstance(processed, dict) else None
                        if isinstance(image_obj, dict) and isinstance(image_obj.get("id"), str):
                            image_id = image_obj.get("id")
                        elif isinstance(processed.get("image_id"), str):  # type: ignore[union-attr]
                            image_id = processed.get("image_id")  # type: ignore[assignment]
                        elif isinstance(processed.get("imageId"), str):  # type: ignore[union-attr]
                            image_id = processed.get("imageId")  # type: ignore[assignment]
                except Exception:
                    pass

            if not image_id:
                raise RuntimeError(
                    "Unable to resolve image_id for annotation upload. Enable wait_for_processing or "
                    "ensure the upload response includes an image id."
                )

            ann_resp = self.annotate_uploaded_image(
                image_id=image_id,
                annotation_path=annotation_path,
                annotation_name=annotation_name,
            )
            # Include the annotation response in the return payload for visibility.
            data["annotation"] = ann_resp  # type: ignore[index]

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
        annotation_directory: Optional[Path | str] = None,
        annotation_suffixes: Sequence[str] = DEFAULT_ANNOTATION_EXTS,
        allow_missing_annotations: bool = False,
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
            annotation_directory: Directory containing per-image annotation files.
            annotation_suffixes: Acceptable extensions for annotation lookups.
            allow_missing_annotations: When False, raise if annotation is missing.

        Returns:
            Mapping of image path to upload metadata.
        """
        directory_path = Path(directory).expanduser()
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Upload directory not found: {directory_path}")

        pattern = "**/*" if recursive else "*"
        uploads: Dict[Path, Dict[str, object]] = {}
        mask_root = Path(mask_directory).expanduser() if mask_directory else None
        annotation_root = (
            Path(annotation_directory).expanduser() if annotation_directory else None
        )

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

            annotation_path = None
            if annotation_root:
                annotation_path = self._match_annotation_path(
                    image_path=image_file,
                    image_root=directory_path,
                    annotation_root=annotation_root,
                    annotation_suffixes=annotation_suffixes,
                )
                if not annotation_path and not allow_missing_annotations:
                    raise FileNotFoundError(
                        f"Annotation not found for {image_file.name} in {annotation_root}"
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
                annotation_path=annotation_path,
            )

        return uploads

    # --------------------------------------------------------------------- #
    # Internal utilities
    # --------------------------------------------------------------------- #
    def annotate_uploaded_image(
        self,
        *,
        image_id: str,
        annotation_path: Path | str,
        annotation_name: Optional[str] = None,
        job_name: Optional[str] = None,
        is_prediction: bool = False,
        overwrite: bool = False,
        labelmap: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Attach an annotation file to an existing uploaded image.

        This calls the Roboflow annotate endpoint:
            POST /dataset/{project}/annotate/{image_id}

        Args:
            image_id: Roboflow image identifier returned after processing an upload.
            annotation_path: Path to the local annotation file (VOC/YOLO/COCO, etc.).
            annotation_name: Optional storage name for the annotation in Roboflow.
            job_name: Optional job name label visible in Roboflow Annotate UI.
            is_prediction: Mark annotations as predictions rather than ground truth.
            overwrite: Overwrite existing annotation if one already exists.
            labelmap: Optional label map when sending COCO-like formats.

        Returns:
            Parsed JSON response of the annotate call.
        """
        p = Path(annotation_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Annotation path does not exist: {p}")

        url = f"{self.API_ROOT}/dataset/{self.dataset_slug}/annotate/{image_id}"
        params = {"api_key": self.api_key, "name": annotation_name or p.name}
        if job_name:
            params["jobName"] = job_name
        if is_prediction:
            params["prediction"] = "true"
        if overwrite:
            params["overwrite"] = "true"

        annotation_string = p.read_text(encoding="utf-8")
        body_payload = {"annotationFile": annotation_string}
        if labelmap is not None:
            body_payload["labelmap"] = labelmap

        try:
            response = self.session.post(
                url,
                params=params,
                data=json.dumps(body_payload),
                headers={"Content-Type": "application/json"},
                timeout=self._timeout_tuple,
            )
        except requests.exceptions.Timeout as exc:  # pragma: no cover - network errors
            raise RuntimeError(
                f"Roboflow annotate timed out after {self.timeout:.0f}s (annotation={p.name}). "
                "Increase --timeout or retry the upload."
            ) from exc
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise RuntimeError(f"Roboflow annotate request failed: {exc}") from exc

        # Inline response handling to allow 409 conflict with a friendly message
        try:
            payload = response.json()
        except Exception:
            payload = None

        if response.status_code not in (200, 409) or not payload:
            body = response.text
            raise RuntimeError(
                f"Roboflow annotate failed: HTTP {response.status_code}; response={body}"
            )

        if response.status_code == 409:
            # Common case: image already annotated
            err = (payload or {}).get("error", {}) if isinstance(payload, dict) else {}
            msg = err.get("message") if isinstance(err, dict) else None
            if isinstance(msg, str) and "already annotated" in msg.lower():
                return {"warn": "already annotated"}
            raise RuntimeError(f"Roboflow annotate conflict: {payload}")

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"Roboflow annotate error: {payload}")
        if isinstance(payload, dict) and not payload.get("success", True):
            raise RuntimeError(f"Roboflow annotate unsuccessful: {payload}")

        return payload if isinstance(payload, dict) else {"result": payload}

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
        return self._match_related_path(
            image_path=image_path,
            image_root=image_root,
            related_root=mask_root,
            candidate_suffixes=mask_suffixes,
        )

    def _match_annotation_path(
        self,
        *,
        image_path: Path,
        image_root: Path,
        annotation_root: Path,
        annotation_suffixes: Sequence[str],
    ) -> Optional[Path]:
        """Resolve an annotation path for ``image_path`` within ``annotation_root``."""
        return self._match_related_path(
            image_path=image_path,
            image_root=image_root,
            related_root=annotation_root,
            candidate_suffixes=annotation_suffixes,
        )

    def _match_related_path(
        self,
        *,
        image_path: Path,
        image_root: Path,
        related_root: Path,
        candidate_suffixes: Sequence[str],
    ) -> Optional[Path]:
        """Generic helper to match a related file (mask/annotation) for an image."""
        try:
            relative = image_path.relative_to(image_root)
        except ValueError:
            relative = Path(image_path.name)

        if relative.parent == Path("."):
            search_dirs = [related_root]
        else:
            search_dirs = [related_root / relative.parent, related_root]

        suffix_candidates = []
        if relative.suffix:
            suffix_candidates.append(relative.suffix)
        for suffix in candidate_suffixes:
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
            try:
                response = self.session.get(
                    status_url,
                    params=params,
                    timeout=self._timeout_tuple,
                )
            except requests.exceptions.Timeout:
                time.sleep(self.poll_interval)
                continue
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Roboflow status polling failed: {exc}") from exc

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


def _load_env_from_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    - Skips lines that are empty or start with '#'.
    - Does not override variables that are already set in the environment.
    - Trims surrounding single or double quotes around values.
    """
    try:
        dotenv_path = Path(path)
        if not dotenv_path.is_file():
            return
        for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if (val.startswith("'") and val.endswith("'")) or (
                val.startswith('"') and val.endswith('"')
            ):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Fail silently; env-file loading is a convenience only.
        return


def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload images to Roboflow Annotate.")
    parser.add_argument("--api-key", help="Roboflow API key (env: ROBOFLOW_API_KEY).")
    parser.add_argument(
        "--workspace",
        help="Roboflow workspace slug (env: ROBOFLOW_WORKSPACE).",
    )
    parser.add_argument(
        "--project",
        help="Roboflow project slug (env: ROBOFLOW_PROJECT).",
    )
    parser.add_argument(
        "--version",
        help="Dataset version to target, e.g. 1 (env: ROBOFLOW_VERSION).",
    )
    parser.add_argument(
        "--path",
        help="File or directory to upload (env: ROBOFLOW_PATH).",
    )
    parser.add_argument("--split", default="train", help="Dataset split tag (default: train).")
    parser.add_argument("--batch", help="Optional batch name for the upload.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP timeout (seconds) for Roboflow API requests (default: 60).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Number of retry attempts for Roboflow API requests (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=None,
        help="Exponential backoff factor between retries in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds to wait between status checks when waiting for processing (default: 2.5).",
    )
    parser.add_argument(
        "--mask",
        help="Mask file to upload alongside --path when uploading a single image.",
    )
    parser.add_argument(
        "--annotation",
        help="Annotation file to upload alongside --path (single image only).",
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
        "--annotation-dir",
        help="Directory containing per-image annotation files that mirror the image filenames.",
    )
    parser.add_argument(
        "--annotation-ext",
        dest="annotation_exts",
        action="append",
        help="Additional annotation extensions to consider (e.g. .xml, .txt).",
    )
    parser.add_argument(
        "--allow-missing-annotations",
        action="store_true",
        help="Skip annotation attachment when a matching file cannot be found.",
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
    # Load .env first so environment fallbacks are populated for this process.
    _load_env_from_file(".env")

    args = _parse_cli_args(argv)

    api_key = _env_or_arg(getattr(args, "api_key", None), "ROBOFLOW_API_KEY")
    workspace = _env_or_arg(getattr(args, "workspace", None), "ROBOFLOW_WORKSPACE")
    project = _env_or_arg(getattr(args, "project", None), "ROBOFLOW_PROJECT")
    version_raw = _env_or_arg(getattr(args, "version", None), "ROBOFLOW_VERSION")
    path_str = _env_or_arg(getattr(args, "path", None), "ROBOFLOW_PATH")

    missing = []
    if not api_key:
        missing.append("ROBOFLOW_API_KEY / --api-key")
    if not workspace:
        missing.append("ROBOFLOW_WORKSPACE / --workspace")
    if not project:
        missing.append("ROBOFLOW_PROJECT / --project")
    if not path_str:
        missing.append("ROBOFLOW_PATH / --path")
    if missing:
        raise ValueError(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Populate .env or pass flags."
        )

    timeout_raw = _env_or_arg(
        str(args.timeout) if getattr(args, "timeout", None) is not None else None,
        "ROBOFLOW_TIMEOUT",
    )
    timeout_value = float(timeout_raw) if timeout_raw else 60.0

    retries_raw = _env_or_arg(
        str(args.retries) if getattr(args, "retries", None) is not None else None,
        "ROBOFLOW_RETRIES",
    )
    retries_value = int(retries_raw) if retries_raw else 3

    retry_backoff_raw = _env_or_arg(
        str(args.retry_backoff)
        if getattr(args, "retry_backoff", None) is not None
        else None,
        "ROBOFLOW_RETRY_BACKOFF",
    )
    retry_backoff_value = float(retry_backoff_raw) if retry_backoff_raw else 1.0

    poll_interval_raw = _env_or_arg(
        str(args.poll_interval)
        if getattr(args, "poll_interval", None) is not None
        else None,
        "ROBOFLOW_POLL_INTERVAL",
    )
    poll_interval_value = float(poll_interval_raw) if poll_interval_raw else 2.5

    annotator = RoboflowAnnotator(
        api_key=api_key,
        workspace=workspace,
        project=project,
        version=version_raw or None,
        timeout=float(timeout_value),
        poll_interval=poll_interval_value,
        max_retries=retries_value,
        retry_backoff=retry_backoff_value,
    )

    candidate_path = Path(path_str).expanduser()
    metadata = json.loads(args.metadata) if args.metadata else None
    mask_extra_exts = []
    if args.mask_exts:
        for ext in args.mask_exts:
            if not ext:
                continue
            cleaned = ext.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            mask_extra_exts.append(cleaned)
    mask_suffixes = tuple(dict.fromkeys([*DEFAULT_MASK_EXTS, *mask_extra_exts]))

    annotation_extra_exts = []
    if args.annotation_exts:
        for ext in args.annotation_exts:
            if not ext:
                continue
            cleaned = ext.strip().lower()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            annotation_extra_exts.append(cleaned)
    annotation_suffixes = tuple(
        dict.fromkeys([*DEFAULT_ANNOTATION_EXTS, *annotation_extra_exts])
    )

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
            annotation_directory=args.annotation_dir,
            annotation_suffixes=annotation_suffixes,
            allow_missing_annotations=args.allow_missing_annotations,
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

        annotation_path = Path(args.annotation).expanduser() if args.annotation else None
        if not annotation_path and args.annotation_dir:
            annotation_root = Path(args.annotation_dir).expanduser()
            annotation_candidate = annotator._match_annotation_path(  # type: ignore[attr-defined]
                image_path=candidate_path,
                image_root=candidate_path.parent,
                annotation_root=annotation_root,
                annotation_suffixes=annotation_suffixes,
            )
            annotation_path = annotation_candidate
            if (
                not annotation_path
                and not args.allow_missing_annotations
                and args.annotation_dir
            ):
                raise FileNotFoundError(
                    f"Annotation not found for {candidate_path.name} in {annotation_root}"
                )

        payload = annotator.upload_image(
            candidate_path,
            split=args.split,
            batch_name=args.batch,
            metadata=metadata,
            assignee_email=args.assignee,
            mask_path=mask_path,
            annotation_path=annotation_path,
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

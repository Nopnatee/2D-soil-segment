"""Minimal helpers to upload images and paired annotations to Roboflow.

The implementation mirrors the public REST documentation at
https://docs.roboflow.com/api-reference/upload and the open-source
`roboflow-python` client so that uploads can include annotations in a
single helper call.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import requests

DEFAULT_IMAGE_EXTS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_ANNOTATION_EXTS: Sequence[str] = (".txt", ".json", ".xml", ".jsonl", ".geojson")


class RoboflowAnnotator:
    """Small wrapper around the Roboflow upload and annotate endpoints."""

    API_ROOT = "https://api.roboflow.com"

    def __init__(
        self,
        api_key: str,
        project: str,
        workspace: Optional[str] = None,
        *,
        version: Optional[int | str] = None,
        timeout: float = 60.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("Roboflow API key is required.")

        project_slug = project.strip().strip("/")
        workspace_slug = workspace.strip().strip("/") if workspace else ""

        if "/" in project_slug:
            inferred_workspace, inferred_project = project_slug.split("/", 1)
            if workspace_slug and workspace_slug != inferred_workspace:
                raise ValueError("Workspace mismatch between --workspace and --project.")
            workspace_slug = workspace_slug or inferred_workspace
            project_slug = inferred_project

        if not project_slug:
            raise ValueError("Roboflow project slug is required.")

        self.api_key = api_key
        self.workspace = workspace_slug
        self.project = project_slug
        self.dataset_slug = project_slug
        self.version = str(version).lstrip("v") if version is not None else None
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    def upload_image(
        self,
        image_path: Path | str,
        *,
        split: str = "train",
        batch_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        annotation_path: Optional[Path | str] = None,
        job_name: Optional[str] = None,
        is_prediction: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, object]:
        """Upload a single image and optional annotation."""
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        url = f"{self.API_ROOT}/dataset/{self.dataset_slug}/upload"
        params = self._build_upload_params(
            file_name=path.name,
            split=split,
            batch=batch_name,
            tags=tags,
        )

        mime_type, _ = mimetypes.guess_type(path.name)
        mime_type = mime_type or "application/octet-stream"

        try:
            with path.open("rb") as image_stream:
                files = {"file": (path.name, image_stream, mime_type)}
                response = self.session.post(
                    url,
                    params=params,
                    files=files,
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise RuntimeError(f"Roboflow upload request failed for {path.name}: {exc}") from exc

        data = self._json_or_raise(response)

        image_id = self._extract_image_id(data)
        if annotation_path:
            if not image_id:
                raise RuntimeError("Upload succeeded but no image_id returned; cannot attach annotation.")
            annotation_payload = self._send_annotation(
                image_id=image_id,
                annotation_path=annotation_path,
                job_name=job_name,
                is_prediction=is_prediction,
                overwrite=overwrite,
            )
            data["annotation"] = annotation_payload  # type: ignore[index]
        return data

    def upload_directory(
        self,
        images_dir: Path | str,
        *,
        annotations_dir: Optional[Path | str] = None,
        annotation_suffixes: Sequence[str] = DEFAULT_ANNOTATION_EXTS,
        split: str = "train",
        batch_name: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        recursive: bool = False,
        skip_missing_annotations: bool = True,
        overwrite_annotations: bool = False,
        is_prediction: bool = False,
    ) -> Dict[Path, Dict[str, object]]:
        """Upload every image inside a directory, pairing annotations by filename."""
        root = Path(images_dir).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(f"Image directory not found: {root}")

        annotation_root = Path(annotations_dir).expanduser() if annotations_dir else None
        iterator = root.rglob("*") if recursive else root.glob("*")

        uploads: Dict[Path, Dict[str, object]] = {}
        for image in sorted(iterator):
            if image.is_dir() or image.suffix.lower() not in DEFAULT_IMAGE_EXTS:
                continue

            annotation_path: Optional[Path] = None
            if annotation_root:
                annotation_path = self._match_annotation(
                    image_path=image,
                    images_root=root,
                    annotation_root=annotation_root,
                    candidate_suffixes=annotation_suffixes,
                )
                if not annotation_path and not skip_missing_annotations:
                    raise FileNotFoundError(f"Missing annotation for {image.name} in {annotation_root}")

            uploads[image] = self.upload_image(
                image,
                split=split,
                batch_name=batch_name,
                tags=tags,
                annotation_path=annotation_path,
                overwrite=overwrite_annotations,
                is_prediction=is_prediction,
            )
        return uploads

    def _build_upload_params(
        self,
        *,
        file_name: str,
        split: str,
        batch: Optional[str],
        tags: Optional[Sequence[str]],
    ) -> Dict[str, object]:
        params: Dict[str, object] = {
            "api_key": self.api_key,
            "name": file_name,
            "split": split,
        }
        if batch:
            params["batch"] = batch
        if tags:
            params["tag"] = list(tags)
        if self.version:
            params["version"] = self.version
        return params

    def _send_annotation(
        self,
        *,
        image_id: str,
        annotation_path: Path | str,
        job_name: Optional[str],
        is_prediction: bool,
        overwrite: bool,
    ) -> Dict[str, object]:
        annotation_file = Path(annotation_path).expanduser()
        if not annotation_file.is_file():
            raise FileNotFoundError(f"Annotation not found: {annotation_file}")

        url = f"{self.API_ROOT}/dataset/{self.dataset_slug}/annotate/{image_id}"
        params = {"api_key": self.api_key, "name": annotation_file.name}
        if self.version:
            params["version"] = self.version
        if job_name:
            params["jobName"] = job_name
        if is_prediction:
            params["prediction"] = "true"
        if overwrite:
            params["overwrite"] = "true"

        payload: Dict[str, object] = {
            "annotationFile": annotation_file.read_text(encoding="utf-8"),
        }

        try:
            response = self.session.post(
                url,
                params=params,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Roboflow annotate request failed for {annotation_file.name}: {exc}") from exc

        return self._handle_annotation_response(response)

    def _match_annotation(
        self,
        *,
        image_path: Path,
        images_root: Path,
        annotation_root: Path,
        candidate_suffixes: Sequence[str],
    ) -> Optional[Path]:
        try:
            relative = image_path.relative_to(images_root)
        except ValueError:
            relative = Path(image_path.name)

        search_dirs = [annotation_root]
        if relative.parent != Path("."):
            search_dirs.insert(0, annotation_root / relative.parent)

        suffixes: list[str] = []
        if image_path.suffix:
            suffixes.append(image_path.suffix)
            lowered = image_path.suffix.lower()
            if lowered not in suffixes:
                suffixes.append(lowered)
        for suffix in candidate_suffixes:
            cleaned = suffix.strip()
            if not cleaned:
                continue
            if not cleaned.startswith("."):
                cleaned = f".{cleaned}"
            cleaned = cleaned.lower()
            if cleaned not in suffixes:
                suffixes.append(cleaned)

        stem = relative.stem
        for base_dir in search_dirs:
            for suffix in suffixes:
                candidate = base_dir / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    def _json_or_raise(self, response: requests.Response) -> Dict[str, object]:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = exc.response.text if exc.response is not None else "No response body."
            raise RuntimeError(f"Roboflow upload failed: {exc}; response={body}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Roboflow upload did not return JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected upload payload type: {type(payload)!r}")
        if payload.get("error"):
            raise RuntimeError(f"Roboflow upload error: {payload}")
        if not payload.get("success", True) and not payload.get("duplicate"):
            raise RuntimeError(f"Roboflow upload unsuccessful: {payload}")

        return payload

    def _handle_annotation_response(self, response: requests.Response) -> Dict[str, object]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Roboflow annotate did not return JSON.") from exc

        if response.status_code == 409:
            error = payload.get("error") if isinstance(payload, dict) else None
            message = error.get("message") if isinstance(error, dict) else None
            if isinstance(message, str) and "already annotated" in message.lower():
                return {"warn": "already annotated"}
            body = response.text
            raise RuntimeError(f"Roboflow annotate conflict: {body}")

        if response.status_code != 200:
            body = response.text
            raise RuntimeError(f"Roboflow annotate failed (status {response.status_code}): {body}")

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected annotate payload type: {type(payload)!r}")
        if payload.get("error"):
            raise RuntimeError(f"Roboflow annotate error: {payload}")
        if not payload.get("success", True):
            raise RuntimeError(f"Roboflow annotate unsuccessful: {payload}")

        return payload

    @staticmethod
    def _extract_image_id(payload: Dict[str, object]) -> Optional[str]:
        image = payload.get("image")
        if isinstance(image, dict) and isinstance(image.get("id"), str):
            return image["id"]
        for key in ("id", "image_id", "imageId"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return None


def _env_or_arg(value: Optional[str], env_key: str) -> str:
    return value or os.getenv(env_key, "")


def _load_env_from_file(path: str = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def _parse_cli_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload images and annotations to Roboflow.")
    parser.add_argument("--api-key", help="Roboflow API key (env: ROBOFLOW_API_KEY).")
    parser.add_argument("--workspace", help="Optional workspace slug (env: ROBOFLOW_WORKSPACE).")
    parser.add_argument("--project", help="Roboflow project slug (env: ROBOFLOW_PROJECT).")
    parser.add_argument("--version", help="Dataset version (env: ROBOFLOW_VERSION).")
    parser.add_argument("--images", type=Path, help="Image file or directory to upload (env: ROBOFLOW_PATH).")
    parser.add_argument("--annotations", type=Path, help="Directory with annotation files matching image stems.")
    parser.add_argument(
        "--annotation-ext",
        dest="annotation_exts",
        nargs="*",
        help="Additional annotation extensions to try (default: txt json xml jsonl geojson).",
    )
    parser.add_argument("--split", default="train", help="Dataset split to target (default: train).")
    parser.add_argument("--batch", help="Optional batch name for the uploads.")
    parser.add_argument("--tags", nargs="*", help="Optional tags to attach to each image.")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds (default: 60).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when uploading folders.")
    parser.add_argument(
        "--no-skip-missing-annotations",
        action="store_true",
        help="Fail if an annotation is missing instead of skipping.",
    )
    parser.add_argument(
        "--overwrite-annotations",
        action="store_true",
        help="Overwrite existing annotations in Roboflow if one already exists.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    _load_env_from_file()
    args = _parse_cli_args(argv)

    api_key = _env_or_arg(args.api_key, "ROBOFLOW_API_KEY")
    project = _env_or_arg(args.project, "ROBOFLOW_PROJECT")
    workspace = _env_or_arg(args.workspace, "ROBOFLOW_WORKSPACE") or None
    version = _env_or_arg(args.version, "ROBOFLOW_VERSION") or None

    images_arg = str(args.images) if args.images else None
    images_target = _env_or_arg(images_arg, "ROBOFLOW_PATH")
    if not api_key or not project or not images_target:
        raise SystemExit("Missing required --api-key, --project, or --images/ROBOFLOW_PATH value.")

    timeout = float(args.timeout) if args.timeout else 60.0
    annotator = RoboflowAnnotator(
        api_key=api_key,
        project=project,
        workspace=workspace,
        version=version or None,
        timeout=timeout,
    )

    tags = args.tags or None
    annotation_suffixes = tuple(args.annotation_exts) if args.annotation_exts else DEFAULT_ANNOTATION_EXTS
    image_path = Path(images_target).expanduser()
    annotation_dir = args.annotations.expanduser() if args.annotations else None

    if image_path.is_dir():
        uploads = annotator.upload_directory(
            image_path,
            annotations_dir=annotation_dir,
            annotation_suffixes=annotation_suffixes,
            split=args.split,
            batch_name=args.batch,
            tags=tags,
            recursive=args.recursive,
            skip_missing_annotations=not args.no_skip_missing_annotations,
            overwrite_annotations=args.overwrite_annotations,
        )
        for uploaded_path, payload in uploads.items():
            status = (
                payload.get("status")
                or payload.get("result")
                or payload.get("success")
                or payload.get("warn")
                or "ok"
            )
            print(f"[{uploaded_path}] -> {status}")
    else:
        annotation_path = None
        if annotation_dir:
            annotation_path = annotator._match_annotation(
                image_path=image_path,
                images_root=image_path.parent,
                annotation_root=annotation_dir,
                candidate_suffixes=annotation_suffixes,
            )
            if not annotation_path and args.no_skip_missing_annotations:
                raise FileNotFoundError(f"Missing annotation for {image_path.name} in {annotation_dir}")

        payload = annotator.upload_image(
            image_path,
            split=args.split,
            batch_name=args.batch,
            tags=tags,
            annotation_path=annotation_path,
            overwrite=args.overwrite_annotations,
        )
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()

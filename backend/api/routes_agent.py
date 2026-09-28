"""Compact, read-only production access for film-making agents.

The UI is intentionally rich; an agent should not need to infer production
state from screenshots or client-side state.  This route is a stable summary
of the same vault records used by the UI.  Paid mutations remain on the
existing quote/approval/generation routes and are explicitly described by the
capability catalog rather than being silently triggered here.
"""

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()
VAULT_DIR = Path(__file__).resolve().parent.parent / "assets"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


def _project_dir(project_id: str) -> Path:
    if not SAFE_ID.fullmatch(project_id):
        raise HTTPException(status_code=400, detail="Invalid project id")
    directory = VAULT_DIR / project_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Project not found")
    return directory


def _load(directory: Path, name: str, fallback: Any) -> Any:
    path = directory / name
    if not path.is_file():
        return fallback
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail=f"Project record {name} is invalid") from error


def _capabilities() -> list[dict[str, Any]]:
    # These routes are already implemented by the application.  The catalog
    # makes their cost and confirmation semantics machine-readable.
    return [
        {"id": "read_project", "method": "GET", "path": "/api/agent/projects/{project_id}/state", "paid": False},
        {"id": "read_shot_package", "method": "GET", "path": "/api/agent/projects/{project_id}/shots/{shot_id}/package", "paid": False},
        {"id": "list_models", "method": "GET", "path": "/api/generate/drivers", "paid": False},
        {"id": "create_asset", "method": "POST", "path": "/api/generate/image", "paid": True, "requires": ["explicit_cost_confirmation"]},
        {"id": "generate_shot", "method": "POST", "path": "/api/shots/video", "paid": True, "requires": ["explicit_cost_confirmation", "project_id", "shot_id"]},
        {"id": "generate_audio", "method": "POST", "path": "/api/audio/job", "paid": True, "requires": ["explicit_cost_confirmation"]},
        {"id": "select_take", "method": "POST", "path": "/api/shots/video/take/select", "paid": False},
        {"id": "render_timeline", "method": "POST", "path": "/api/timeline/{project_id}/render", "paid": False},
        {"id": "export_edl", "method": "GET", "path": "/api/export/{project_id}/edl", "paid": False},
    ]


def _state(project_id: str) -> dict[str, Any]:
    directory = _project_dir(project_id)
    project = _load(directory, "project.json", {})
    scenes = _load(directory, "scenes.json", [])
    shots = _load(directory, "shots.json", [])
    assets = _load(directory, "assets.json", [])
    timeline = _load(directory, "timeline.json", {"tracks": [], "clips": []})
    ready_videos = [shot for shot in shots if shot.get("video_clip_path")]
    return {
        "schema_version": "ai-movie-studio-agent/v1",
        "project": {
            "id": project_id,
            "name": project.get("name", project_id),
            "description": project.get("description", ""),
            "created_at": project.get("created_at"),
        },
        "production": {
            "scenes": scenes,
            "shots": shots,
            "assets": assets,
            "timeline": timeline,
        },
        "summary": {
            "scene_count": len(scenes),
            "shot_count": len(shots),
            "asset_count": len(assets),
            "shots_with_keyframes": sum(bool(shot.get("frame_image_path")) for shot in shots),
            "shots_with_video": len(ready_videos),
            "ready_for_video_generation": [shot.get("id") for shot in shots if shot.get("frame_image_path") and not shot.get("video_clip_path")],
        },
    }


@router.get("/capabilities")
async def agent_capabilities():
    return {
        "protocol": "ai-movie-studio-agent/v1",
        "rule": "Read project state before every mutation. Never submit a paid action without an explicit displayed quote and confirmation.",
        "actions": _capabilities(),
    }


@router.get("/projects/{project_id}/state")
async def project_state(project_id: str):
    return _state(project_id)


@router.get("/projects/{project_id}/shots/{shot_id}/package")
async def shot_package(project_id: str, shot_id: str):
    if not SAFE_ID.fullmatch(shot_id):
        raise HTTPException(status_code=400, detail="Invalid shot id")
    state = _state(project_id)
    shot = next((item for item in state["production"]["shots"] if item.get("id") == shot_id), None)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    scene = next((item for item in state["production"]["scenes"] if item.get("id") == shot.get("scene_id")), None)
    asset_ids = {item.get("asset_id") for item in shot.get("assets", []) if item.get("asset_id")}
    assets = [asset for asset in state["production"]["assets"] if asset.get("id") in asset_ids]
    return {
        "schema_version": "ai-movie-studio-agent/v1",
        "project": state["project"],
        "scene": scene,
        "shot": shot,
        "resolved_assets": assets,
        "generation_gate": {
            "has_keyframe": bool(shot.get("frame_image_path")),
            "has_selected_video": bool(shot.get("video_clip_path")),
            "requires_explicit_cost_confirmation": True,
        },
    }

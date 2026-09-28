# Agent production API

`GET /api/agent/projects/{project_id}/state` returns the complete persisted
production graph in one response: project metadata, scenes, shots, assets and
timeline, plus a small readiness summary.

`GET /api/agent/projects/{project_id}/shots/{shot_id}/package` returns the
single shot, its scene and its resolved asset records. Use it immediately
before preparing that shot so an agent does not repeatedly read the web UI.

`GET /api/agent/capabilities` lists the existing action routes and identifies
which ones are paid. The API never exposes secrets. A paid action must first
show its quote and receive explicit confirmation; generation inputs must be
read from the production graph rather than inferred from a screenshot.

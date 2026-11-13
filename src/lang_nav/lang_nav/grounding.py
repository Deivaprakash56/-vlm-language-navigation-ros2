"""Grounding a natural-language command into a navigation target.

Pure Python (no ROS imports) so it can be unit-tested and reused offline.

Three kinds of target are supported:
  * landmark     - one of the labeled places in the map YAML
  * image_point  - a pixel (u, v) in the current camera image; the ROS node
                   back-projects it with the depth image to a 3D map point
  * relative     - a motion relative to the robot ("go forward 1 m", "turn left")
"""
from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

KINDS = ('landmark', 'image_point', 'relative', 'none')


@dataclass
class Landmark:
    name: str
    description: str
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class LandmarkMap:
    frame_id: str
    landmarks: List[Landmark] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> 'LandmarkMap':
        with open(path) as f:
            data = yaml.safe_load(f)
        lms = [Landmark(name=str(d['name']), description=str(d.get('description', '')),
                        x=float(d['x']), y=float(d['y']), yaw=float(d.get('yaw', 0.0)))
               for d in data.get('landmarks', [])]
        return cls(frame_id=data.get('frame_id', 'map'), landmarks=lms)

    def get(self, name: str) -> Optional[Landmark]:
        for lm in self.landmarks:
            if lm.name == name:
                return lm
        return None

    def as_prompt_text(self) -> str:
        return '\n'.join(f'- {lm.name}: {lm.description}' for lm in self.landmarks)


@dataclass
class GroundingResult:
    kind: str
    landmark: str = ''
    pixel_u: int = -1
    pixel_v: int = -1
    forward_m: float = 0.0
    left_m: float = 0.0
    turn_deg: float = 0.0
    reason: str = ''

    def validate(self, lmap: LandmarkMap, image_wh: Optional[tuple] = None) -> 'GroundingResult':
        """Return self if usable, otherwise a 'none' result explaining why."""
        if self.kind not in KINDS:
            return GroundingResult('none', reason=f'unknown kind {self.kind!r}')
        if self.kind == 'landmark' and lmap.get(self.landmark) is None:
            return GroundingResult('none', reason=f'unknown landmark {self.landmark!r}')
        if self.kind == 'image_point':
            if image_wh is None:
                return GroundingResult('none', reason='image_point chosen but no image available')
            w, h = image_wh
            if not (0 <= self.pixel_u < w and 0 <= self.pixel_v < h):
                return GroundingResult('none', reason=f'pixel ({self.pixel_u},{self.pixel_v}) outside {w}x{h}')
        if self.kind == 'relative':
            if abs(self.forward_m) > 5.0 or abs(self.left_m) > 5.0:
                return GroundingResult('none', reason='relative move larger than 5 m safety limit')
        return self


# --------------------------------------------------------------------------- #
# Keyword backend (offline, deterministic) - good for tests and no-API-key runs
# --------------------------------------------------------------------------- #
_NUM = r'(\d+(?:\.\d+)?)'


class KeywordGrounder:
    """Cheap rule-based grounding. Handles landmark names and simple relative moves."""

    def __init__(self, lmap: LandmarkMap):
        self.lmap = lmap

    _STOP = {'the', 'a', 'an', 'to', 'of', 'go', 'please', 'next', 'area', 'over', 'drive', 'head', 'and'}

    @classmethod
    def _tokens(cls, text: str) -> set:
        text = text.lower().replace('_', ' ')
        text = re.sub(r'\b(north|south)(east|west)\b', r'\1 \2', text)  # "northeast" -> "north east"
        return set(re.findall(r'[a-z]+', text)) - cls._STOP

    def ground(self, command: str, image_bgr=None) -> GroundingResult:
        cmd = command.lower().strip()

        # debug syntax to exercise the depth back-projection path without a VLM: "@pixel 320 300"
        px = re.match(r'@pixel\s+(\d+)\s+(\d+)', cmd)
        if px:
            return GroundingResult('image_point', pixel_u=int(px.group(1)), pixel_v=int(px.group(2)),
                                   reason='debug pixel')

        # relative motions
        # "back" only counts as a motion when followed by a distance ("go back 0.5 m"), so
        # "take me back home" still resolves to the landmark
        m = re.search(r'(forward|ahead|backward|reverse|back(?=\s*\d))\s*(?:by\s*)?' + _NUM + r'?\s*(m|meter|metre)?',
                      cmd)
        t = re.search(r'turn\s+(left|right|around)\s*(?:by\s*)?' + _NUM + r'?', cmd)
        if m or t:
            fwd = turn = 0.0
            if m:
                dist = float(m.group(2)) if m.group(2) else 1.0
                fwd = -dist if m.group(1).startswith(('back', 'rev')) else dist
            if t:
                if t.group(1) == 'around':
                    turn = 180.0
                else:
                    ang = float(t.group(2)) if t.group(2) else 90.0
                    turn = ang if t.group(1) == 'left' else -ang
            return GroundingResult('relative', forward_m=fwd, turn_deg=turn, reason='keyword relative move')

        # landmark by token overlap with name (weighted) + description
        ctoks = self._tokens(cmd)
        best, best_score = None, 0.0
        for lm in self.lmap.landmarks:
            ntoks = self._tokens(lm.name)
            dtoks = self._tokens(lm.description)
            score = 3.0 * len(ctoks & ntoks) / max(len(ntoks), 1) + 0.5 * len(ctoks & dtoks)
            if score > best_score:
                best, best_score = lm, score
        if best is not None and best_score >= 1.5:
            return GroundingResult('landmark', landmark=best.name,
                                   reason=f'keyword match score {best_score:.2f}')
        return GroundingResult('none', reason='no landmark or motion keyword matched')


# --------------------------------------------------------------------------- #
# Claude vision backend
# --------------------------------------------------------------------------- #
OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'kind': {'type': 'string', 'enum': list(KINDS)},
        'landmark': {'type': 'string', 'description': 'Exact landmark name, or "" if kind != landmark'},
        'pixel_u': {'type': 'integer', 'description': 'Column of target pixel, -1 if unused'},
        'pixel_v': {'type': 'integer', 'description': 'Row of target pixel, -1 if unused'},
        'forward_m': {'type': 'number'},
        'left_m': {'type': 'number'},
        'turn_deg': {'type': 'number', 'description': 'CCW positive'},
        'reason': {'type': 'string'},
    },
    'required': ['kind', 'landmark', 'pixel_u', 'pixel_v', 'forward_m', 'left_m', 'turn_deg', 'reason'],
    'additionalProperties': False,
}

SYSTEM_PROMPT = """You ground natural-language navigation commands for a ground robot (TurtleBot3).
Pick exactly one target kind:
- "landmark": the command refers to a named place. Use the exact name from the landmark list.
- "image_point": the command refers to something visible in the camera image (an object, a gap,
  "that box", "the pillar on the left"). Give the pixel (pixel_u = column, pixel_v = row) on the
  FLOOR directly in front of / at the base of the target, so a depth lookup lands on a reachable spot.
- "relative": a motion relative to the robot. forward_m (+ ahead), left_m (+ left), turn_deg (+ CCW).
- "none": the command is unsafe, ambiguous, or impossible. Explain in reason.
Prefer "landmark" when both a landmark and an image target fit. Unused numeric fields are -1 for
pixels and 0 for motions; unused landmark is "". Keep reason to one short sentence."""


class ClaudeGrounder:
    """Grounding with Claude (vision + structured JSON output)."""

    def __init__(self, lmap: LandmarkMap, model: str = 'claude-opus-5-5', effort: str = 'low',
                 timeout_s: float = 30.0):
        import anthropic  # imported lazily so the keyword backend works without the SDK
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(timeout=timeout_s)
        self.lmap = lmap
        self.model = model
        self.effort = effort

    def _content(self, command: str, image_bgr) -> list:
        content = []
        if image_bgr is not None:
            import cv2
            ok, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                h, w = image_bgr.shape[:2]
                content.append({'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/jpeg',
                    'data': base64.standard_b64encode(buf.tobytes()).decode('ascii')}})
                content.append({'type': 'text', 'text': f'The image above is the robot camera view, {w}x{h} pixels.'})
        content.append({'type': 'text', 'text': (
            f'Landmarks (map frame "{self.lmap.frame_id}"):\n{self.lmap.as_prompt_text()}\n\n'
            f'Command: {command}')})
        return content

    def ground(self, command: str, image_bgr=None) -> GroundingResult:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': self._content(command, image_bgr)}],
            output_config={'effort': self.effort,
                           'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}},
            # On a safety decline, the API re-runs the request on a fallback model.
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
        )
        if response.stop_reason == 'refusal':
            return GroundingResult('none', reason='model declined the request')
        if response.stop_reason == 'max_tokens':
            return GroundingResult('none', reason='model output truncated')
        text = next((b.text for b in response.content if b.type == 'text'), None)
        if text is None:
            return GroundingResult('none', reason='no text block in response')
        data = json.loads(text)
        return GroundingResult(**{k: data[k] for k in OUTPUT_SCHEMA['required']})


def make_grounder(backend: str, lmap: LandmarkMap, **kwargs):
    if backend == 'claude':
        return ClaudeGrounder(lmap, **kwargs)
    if backend == 'keyword':
        return KeywordGrounder(lmap)
    raise ValueError(f'unknown backend {backend!r} (use "claude" or "keyword")')


def yaw_to_quat(yaw: float):
    """(x, y, z, w) quaternion for a rotation about z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

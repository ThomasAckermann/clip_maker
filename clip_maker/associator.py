"""
Action-to-player association.

Given:
  - A list of action events (frame index + label), from the ActionSpotter
  - Ball detections (frame index + x/y), from the BallTracker
  - Player detections (frame index + track_id + bbox), from the PlayerTracker

Associates each action event with the player track whose bounding-box center
was closest to the ball at (or nearest to) the action frame.

Returns the same action dicts enriched with a `player_track_id` key.
If no player detections are available for an action frame, `player_track_id`
is set to None.
"""

from __future__ import annotations

import math
from typing import Any

from .tracker import Detection as BallDetection
from .player_tracker import PlayerDetection


def associate_actions_to_players(
    actions: list[dict[str, Any]],
    ball_detections: list[BallDetection | None],
    player_detections: list[PlayerDetection],
) -> list[dict[str, Any]]:
    """
    Enrich each action dict with a `player_track_id` field.

    Args:
        actions:           List of action dicts with at least a `frame` key.
                           (As returned by ActionSpotter.spot() → Event.to_dict())
        ball_detections:   Per-frame ball detections for the *clip* (not the full match).
                           Index 0 = clip frame 0. May contain None for undetected frames.
        player_detections: Flat list of PlayerDetection for the clip.

    Returns:
        New list of action dicts — same content, plus `player_track_id`.
    """
    # Index player detections by frame for fast lookup
    player_by_frame: dict[int, list[PlayerDetection]] = {}
    for pd in player_detections:
        player_by_frame.setdefault(pd.frame_idx, []).append(pd)

    enriched: list[dict[str, Any]] = []
    for action in actions:
        action_frame: int = action["frame"]
        track_id = _find_nearest_player(
            action_frame=action_frame,
            ball_detections=ball_detections,
            player_by_frame=player_by_frame,
        )
        enriched.append({**action, "player_track_id": track_id})

    return enriched


def _find_nearest_player(
    action_frame: int,
    ball_detections: list[BallDetection | None],
    player_by_frame: dict[int, list[PlayerDetection]],
) -> int | None:
    """
    Find the track_id of the player closest to the ball at `action_frame`.

    If ball position is unavailable at the exact frame, searches up to
    BALL_SEARCH_RADIUS frames in either direction. If no ball position is
    found at all, falls back to the player closest to the frame centre.
    """
    BALL_SEARCH_RADIUS = 15  # frames — ~0.5 s at 30 fps

    ball = _nearest_ball(ball_detections, action_frame, BALL_SEARCH_RADIUS)

    # Prefer exact frame, then expand outward if no players found there
    PLAYER_SEARCH_RADIUS = 5
    players = _players_near_frame(player_by_frame, action_frame, PLAYER_SEARCH_RADIUS)
    if not players:
        return None

    if ball is not None:
        # Closest player center to ball position
        def dist_to_ball(pd: PlayerDetection) -> float:
            return math.hypot(pd.cx - ball.x, pd.cy - ball.y)

        return min(players, key=dist_to_ball).track_id
    else:
        # No ball found — fall back: player whose bbox centre is nearest
        # to the horizontal midpoint of the first player's frame (heuristic)
        # Just return the most frequently seen track near this frame.
        track_counts: dict[int, int] = {}
        for pd in players:
            track_counts[pd.track_id] = track_counts.get(pd.track_id, 0) + 1
        return max(track_counts, key=lambda t: track_counts[t])


def _nearest_ball(
    detections: list[BallDetection | None],
    frame: int,
    radius: int,
) -> BallDetection | None:
    """Return the ball detection closest in time to `frame`, within `radius` frames."""
    # Check exact frame first
    if 0 <= frame < len(detections) and detections[frame] is not None:
        return detections[frame]

    best: BallDetection | None = None
    best_dist = radius + 1

    for offset in range(1, radius + 1):
        for candidate_frame in (frame - offset, frame + offset):
            if 0 <= candidate_frame < len(detections):
                det = detections[candidate_frame]
                if det is not None and offset < best_dist:
                    best = det
                    best_dist = offset
                    break  # inner loop — found at this offset distance
        if best_dist <= offset:
            break

    return best


def _players_near_frame(
    player_by_frame: dict[int, list[PlayerDetection]],
    frame: int,
    radius: int,
) -> list[PlayerDetection]:
    """Collect all player detections within `radius` frames of `frame`."""
    result: list[PlayerDetection] = []
    for offset in range(radius + 1):
        for candidate in (frame - offset, frame + offset) if offset > 0 else (frame,):
            result.extend(player_by_frame.get(candidate, []))
        if result:
            break  # return detections at the closest available frame
    return result

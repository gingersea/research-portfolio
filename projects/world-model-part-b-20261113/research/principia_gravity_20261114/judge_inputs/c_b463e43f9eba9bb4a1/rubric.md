# Fixed gravity-relation visual rubric v1

Judge only your assigned anonymous case. Read only payload.json, this rubric and
the listed frame-sheet images. Do not inspect other directories, experiment code,
private labels, measurements, other cases, repository documents or session history.
Use view_image on EVERY listed sheet. Do not perform computer vision with code,
query another model, infer labels from filenames or ask another agent.
Save your FIRST result at the assigned output path; do not revise it after feedback.

These are controlled rendered sequences of two equal-radius, same-material balls,
initially held at rest over the same floor, in the same image plane. Green fixed
fiducials belong to the apparatus. The intended test is the initial free fall only:
if both balls release together, constant common gravity predicts
h_left/h_right = (T_left/T_right)^2. Keep left/right identity consistent. Height is
the initial center's vertical distance to its eventual floor-contact center;
T is time from release to first floor contact, not the video's entire duration.
Frame index and nominal timestamp are printed beneath every frame. Ball colors
only identify objects. The visible setup is schematic; flat rendering is not a
physics failure. Do not score visual polish or text-prompt adherence.

First check whether this test is applicable from the observed sequence:
- Both objects and relevant release/contact events must be visible.
- Releases must be synchronized closely enough to compare the initial falls.
- The camera must be stationary, supported by the fixed fiducials/background.
- If an object is hidden/lost, a contact has not happened before the clip ends,
  a camera moves, or releases differ visibly, return ABSTAIN. Such a case can be
  physically possible; it is not automatically an anti-physics failure.
- If the images do not let you determine admissibility or the relation, ABSTAIN.

With sufficient evidence, PASS means the common-gravity relation is consistent
within visual timing/position uncertainty. FAIL means a clearly incompatible
height/flight-time relation under the admissible setup. Do not require equal
arrival times for different drop heights; do not declare failure from speed alone.
Keep any uncertain event range broad; do not invent unseen events.

Return ONLY one valid JSON object with exactly these fields:
{
  "case_id": "assigned ID",
  "verdict": "PASS or FAIL or ABSTAIN",
  "reason": "brief observation-grounded explanation",
  "events": {
    "left_release_frame_range": [0, 1],
    "right_release_frame_range": [0, 1],
    "left_contact_frame_range": null,
    "right_contact_frame_range": null
  },
  "assumptions": {
    "synchronized_release": "supported or violated or unclear",
    "static_camera": "supported or violated or unclear",
    "both_falls_observed": "supported or violated or unclear"
  },
  "sheets_viewed": 4
}
Example values above illustrate schema only, not this case's events or verdict.
Use null for any event you cannot localize. Use actual input sheet count. All frame
ranges are inclusive integer indices in the given sequence; no confidence score.

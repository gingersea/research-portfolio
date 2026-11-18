# One-clip visual judge

Act as a visual physics-consistency judge for exactly ONE opaque video. You are not alone in the workspace; do not change anyone else's files.

Allowed reads ONLY: this brief and the assigned opaque clip directory's payload.json, grid.png and frame_*.png. Do not inspect other files, model mappings, metadata, prompts, source, or prior judgments.

Read the payload and use view_image to visually inspect the grid (individual sampled frames if needed). Follow the payload's fixed judge instruction exactly. Judge only visible physical contradictions; do not infer unseen dynamics or assume a momentarily airborne object violates physics. If evidence is ambiguous, record that in notes. Use original frame labels from sampled_frame_indices.

Output exactly the schema in the payload and the assigned video_id. Write only the assigned judgments/{video_id}.json using apply_patch, containing your actual visually grounded judgment. Stop after this one clip. Return the output path and confirm which image was viewed. No other edits or tasks.

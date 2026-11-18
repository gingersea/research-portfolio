# Acquisition-only early stop, before model inference and scores

V2 selection is unchanged: first12 complete eligible distinct-source clips in
archive order, within a maximum first3GiB. Do not continue downloading unused
prefix bytes once those same first12 clips are already certified complete.
Completion requires observing the next directory boundary; the last partial
clip is always excluded from eligibility. The selected12 do not depend on
later archive bytes. This reduces unnecessary source transfer without changing
any sample, model, threshold, annotation rule or endpoint from V2.

Record the actually acquired contiguous prefix length and hash, all candidate
clip metadata, and retained extra completed/partial range files. Original raw
bytes are never altered. If the3GiB limit is reached first, V2's fewer-case rule
still applies. This amendment and eventual sample freeze precede all inference.

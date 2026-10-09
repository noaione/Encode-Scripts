from boane_helpers import collect_samples, read_sample, save_samples

bd_clip, dcp_clip = read_sample("darkscene", 100)
# cut 1s in for bd, sine they have black frames at the start for logotype
# bd_clip = bd_clip[24:]

frame_select = [
    7,
    204,
    236,
    334,
    847,
    1211,
    1301,
]

# Choose corresponding frames from an aligned shot.
dcp_rgb, bd_rgb = collect_samples(
    dcp_clip,
    bd_clip,
    frames=frame_select,
)

save_samples("boane-samples-darkscene.npz", dcp_rgb=dcp_rgb, bd_rgb=bd_rgb)

# bd_clip.set_output(0)
# dcp_clip.set_output(1)

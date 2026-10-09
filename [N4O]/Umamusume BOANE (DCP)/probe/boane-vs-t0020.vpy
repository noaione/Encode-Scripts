from boane_helpers import collect_samples, read_sample, save_samples

bd_clip, dcp_clip = read_sample("t0020", 62)

# Choose corresponding frames from an aligned shot.
dcp_rgb, bd_rgb = collect_samples(
    dcp_clip,
    bd_clip,
    frames=[
        850,
        294,
        660,
        95,
        266,
        410,
        362,
        1130,
        1323,
    ],
)

save_samples("boane-samples-t0020.npz", dcp_rgb=dcp_rgb, bd_rgb=bd_rgb)
# cut 1s in for bd, sine they have black frames at the start for logotype
# bd_clip = bd_clip[24:]

# bd_clip.set_output(0)
# dcp_clip.set_output(1)
# dcp_fix.set_output(2)

import havsfunc as haf
import n4ofunc as nao
import insaneAA as iaa
import vapoursynth as vs
from finedehalo import fine_dehalo
from stgfunc import adaptive_grain
from vapoursynth import core
from vardautomation import VPath
from vsscale import ssim_downsample
from vstools import get_w, depth, insert_clip

from encode_common import open_source, dlisr_upscale, deband_texmask, start_encode


def mean_deband(clip: vs.VideoNode, clip_mask: vs.VideoNode | None = None):
    texmask = clip_mask or deband_texmask(clip, 1)
    debanders = [
        core.neo_f3kdb.Deband(clip, range=16, y=32, grainy=16, grainc=12),
        core.neo_f3kdb.Deband(clip, range=16, y=28, cb=24, grainy=24, grainc=12),
        core.placebo.Deband(clip, radius=10, threshold=4.5, iterations=2, grain=4),
    ]
    return core.std.MaskedMerge(core.average.Mean(debanders), clip, texmask)


CURRENT_FILE = VPath(__file__)

source = open_source("00004.m2ts", CURRENT_FILE, trims=(12, -12))
src = depth(source.clip_cut, 16)

# Timing
ED_CREDS_START = 4639
END_CARDS_START = 5839

main_parts = src[:ED_CREDS_START]
ed_parts = src[ED_CREDS_START : ED_CREDS_START + 1200]
end_cards = src[END_CARDS_START:]

# Rescale (also dehalo, and AA)
dn_mask = nao.simple_native_mask(src, get_w(862), 862, iter_max=5, no_resize=True)
desc_main = iaa.revert_upscale(
    src,
    0.99,
    kernel="bicubic",
    descale_width=get_w(862),
    descale_height=862,
    bicubic_b=0.2,
    bicubic_c=0.5,
    dehalo=False,
)
upscale_back = ssim_downsample(dlisr_upscale(desc_main, 2), height=1080)
desc_fin = core.std.MaskedMerge(upscale_back, src, dn_mask)

# extra dehalo
fin_dh = fine_dehalo(desc_fin, rx=2.15, ry=2.25, darkstr=0.85, brightstr=1.75)
texmask = deband_texmask(fin_dh, 1)
# deband
dbd_r = mean_deband(fin_dh, texmask)
# aggresive denoise
degr_r = haf.SMDegrain(dbd_r, tr=2, thSAD=250, RefineMotion=True)
degr = core.std.MaskedMerge(degr_r, dbd_r, texmask)

# fin_ed = insert_clip(src, ed_dehalo, ED_CREDS_START)

# end cards
# pls dont fuck it up
endfd = fine_dehalo(end_cards)
flt_mix = insert_clip(degr, insert=endfd, start_frame=END_CARDS_START)

# grainset
flt_grain = adaptive_grain(flt_mix, strength=0.3, luma_scaling=8)


# Output
# source.clip_cut.set_output(0)
# desc_test.set_output(0)
# desc_main.set_output(1)
# fin_ed.set_output(0)
# ed_parts.set_output(0)
# ed_debic.set_output(1)
# ed_delanc.set_output(2)
# ed_dehalo.set_output(1)
if __name__ == "__main__":
    start_encode(source, flt_grain)
else:
    # for idx, clip in enumerate([desc_fin, flt_mix, degr, dbd_r, flt_grain, texmask]):
    #     clip.set_output(idx)
    # FOR COMP
    debug_info = "Frame {n} of {total_abs}\n" "Picture Type: {fp_pict_type}\n" "{width}x{height} ({ar})"
    nao.debug_clip(src, f"Source\n{debug_info}").set_output(0)
    nao.debug_clip(mean_deband(src), f"Mean Deband\n{debug_info}").set_output(1)
    # nao.debug_clip(texmask, f"Texture Mask\n{debug_info}").set_output(2)
    # nao.debug_clip(flt_grain, f"Filtered\n{debug_info}").set_output(1)

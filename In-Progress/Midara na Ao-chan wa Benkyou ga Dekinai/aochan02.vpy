import vapoursynth as vs
from vapoursynth import core
import fvsfunc as fvf
import kagefunc as kgf
import n4ofunc as nao
from vsutil import insert_clip, iterate, get_y

core.max_cache_size = 12555

abema = nao.src('Midara na Ao-chan wa Benkyou ga Dekinai - 02 (AbemaTV 1080p).mkv', depth=16)
vrv = nao.src('Midara na Ao-chan wa Benkyou ga Dekinai - 02 (VRV 1080p).mkv', depth=16, trims=[24, 17621])

src = insert_clip(
	abema, 
	vrv[521:649+1], 
	521
)

descaled = nao.descale(
	src, 
	1440, 
	810, 
	kernel='spline36', 
	expandN=4, 
	inflateN=3, 
	yuv444=True, 
	masked=True
)

### OP
op_part = src[521:1959+1]
op_resize = core.resize.Spline16(op_part, 1440, 810, format=nao.get_444(descaled))
op_hard_deband = core.f3kdb.Deband(op_resize, 16, 70, 55, 55, 15, 0, output_depth=16)
op_harder_deband = core.f3kdb.Deband(op_resize, 12, 75, 65, 65, 20, 0, output_depth=16)
op_debanded = insert_clip(op_resize, op_hard_deband[0:128+1], 0)
op_debanded = insert_clip(op_debanded, op_harder_deband[673:696+1], 673)
op_debanded = insert_clip(op_debanded, op_harder_deband[1099:1146+1], 1099)
op_debanded = insert_clip(
	op_debanded, 
	core.grain.Add(
		op_debanded, 
		1, 
		1
	)
	[673:696+1], 
	673
) # Fuck banding

### ED
ed_part = src[15920:17357+1]
ed_resize = core.resize.Spline16(ed_part, 1440, 810, format=nao.get_444(descaled))
ed_debanded = core.f3kdb.Deband(ed_resize, 12, 35, 20, 20, 10, 0, output_depth=16)

### Other filter (Except OP and ED)
sobel_edge = iterate(core.std.Sobel(get_y(descaled)), core.std.Inflate, 2)

adaptmask_dark = kgf.adaptive_grain(descaled, luma_scaling=30, show_mask=True)
adaptmask_light = kgf.adaptive_grain(descaled, luma_scaling=5, show_mask=True).std.Invert()
adaptmask_dark = core.std.Expr([adaptmask_dark, sobel_edge], 'x y -')
adaptmask_light = core.std.Expr([adaptmask_light, sobel_edge], 'x y -')

normal_deband = core.f3kdb.Deband(descaled, 12, 45, 35, 35, 15, 0, output_depth=16)
much_harder_deband = core.f3kdb.Deband(descaled, 16, 75, 60, 60, 20, 0, output_depth=16)

vid = core.std.MaskedMerge(descaled, normal_deband, adaptmask_light)
vid = core.std.MaskedMerge(descaled, much_harder_deband, adaptmask_dark)

### Merge Everything
vid = insert_clip(vid, op_debanded, 521)
vid = insert_clip(vid, ed_debanded, 15920)
vid = kgf.adaptive_grain(vid, 0.25)
vid = fvf.Depth(vid, 10)

vid.set_output()
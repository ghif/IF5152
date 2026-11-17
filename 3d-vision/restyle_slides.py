#!/usr/bin/env python3
"""
Restyle IF5152/3d-vision/3d-vision_2025.pdf with Institut Teknologi Bandung (ITB)
color theme and visual identity:
- ITB Deep Cobalt Blue (#005AAB, PANTONE 2728 CVC) and Metallic Gold (#D4AF37).
- Replaces Stanford Title Slide (Page 1) with an ITB Title Slide + dedicated Credits Slide
  honoring Prof. Jiajun Wu and Stanford CS231n (Spring 2025), plus contributing slide authors.
- Scales original content pages (Pages 2–105) to 91% centered on an ITB canvas so bottom-edge
  slide credits, citations, and full-bleed visuals are 100% preserved without overlap.
- Appends an ITB Key Takeaways / Closing Slide.
"""

import os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pypdf
from pypdf import Transformation

WIDTH_PT = 960.0
HEIGHT_PT = 540.0
ITB_BLUE = '#005AAB'       # Deep Cobalt Blue (PANTONE 2728 CVC)
ITB_GOLD = '#D4AF37'       # Metallic Gold accent
TOP_BLUE_H = 6.0
TOP_GOLD_H = 1.5
FOOTER_BAR_H = 32.0
FOOTER_GOLD_H = 1.5

SCALE = 0.91
TX = (WIDTH_PT - WIDTH_PT * SCALE) / 2.0  # 43.2 pt
AVAIL_H = HEIGHT_PT - (TOP_BLUE_H + TOP_GOLD_H) - (FOOTER_BAR_H + FOOTER_GOLD_H)  # 499.0 pt
TY = (FOOTER_BAR_H + FOOTER_GOLD_H) + (AVAIL_H - HEIGHT_PT * SCALE) / 2.0         # 37.3 pt


def make_page_from_fig(fig):
    buf = BytesIO()
    fig.savefig(buf, format='pdf', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return pypdf.PdfReader(buf).pages[0]


def draw_itb_chrome(ax, slide_num, total_slides, right_label=None):
    """Draws the ITB top accent bar and bottom footer bar on a 960x540 axis."""
    # Top brand bar: ITB Cobalt Blue + Gold
    ax.add_patch(plt.Rectangle((0, HEIGHT_PT - TOP_BLUE_H), WIDTH_PT, TOP_BLUE_H,
                               facecolor=ITB_BLUE, edgecolor='none', zorder=10))
    ax.add_patch(plt.Rectangle((0, HEIGHT_PT - TOP_BLUE_H - TOP_GOLD_H), WIDTH_PT, TOP_GOLD_H,
                               facecolor=ITB_GOLD, edgecolor='none', zorder=10))

    # Bottom bar: ITB Cobalt Blue + Gold accent
    ax.add_patch(plt.Rectangle((0, 0), WIDTH_PT, FOOTER_BAR_H,
                               facecolor=ITB_BLUE, edgecolor='none', zorder=10))
    ax.add_patch(plt.Rectangle((0, FOOTER_BAR_H), WIDTH_PT, FOOTER_GOLD_H,
                               facecolor=ITB_GOLD, edgecolor='none', zorder=10))

    ax.text(30, FOOTER_BAR_H / 2.0, 'Institut Teknologi Bandung  •  IF5152: 3D Vision',
            color='white', fontsize=11.5, fontweight='bold', fontfamily='sans-serif',
            verticalalignment='center', zorder=11)

    label = right_label if right_label is not None else f'{slide_num} / {total_slides}'
    ax.text(WIDTH_PT - 30, FOOTER_BAR_H / 2.0, label,
            color='white', fontsize=11.5, fontfamily='sans-serif',
            horizontalalignment='right', verticalalignment='center', zorder=11)


def create_title_slide(total_slides):
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_PT)
    ax.set_ylim(0, HEIGHT_PT)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 0), WIDTH_PT, HEIGHT_PT, facecolor='#FFFFFF', edgecolor='none'))
    draw_itb_chrome(ax, 1, total_slides)

    # Institution header
    ax.text(65, 475, 'INSTITUT TEKNOLOGI BANDUNG', fontsize=14.5, fontweight='bold',
            color=ITB_BLUE, fontfamily='sans-serif')
    ax.text(65, 452, 'Program Studi Magister Informatika / Sains Data — IF5152 Computer Vision',
            fontsize=12.5, color='#555555', fontfamily='sans-serif')

    # Dividing thin rule
    ax.add_patch(plt.Rectangle((65, 434), 830, 1.2, facecolor='#E2E8F0', edgecolor='none'))

    # Main Title
    ax.text(65, 345, '3D Vision', fontsize=44, fontweight='bold',
            color='#0F172A', fontfamily='sans-serif')

    # Subtitle
    ax.text(65, 290, '3D Representations  •  Multi-View & Voxel CNNs  •  PointNets\nImplicit Surfaces & Neural Radiance Fields (NeRF)  •  3D Gaussian Splatting',
            fontsize=16, color=ITB_BLUE, fontfamily='sans-serif', linespacing=1.45)

    # Attribution / Credit Box
    ax.add_patch(plt.Rectangle((65, 92), 830, 138, facecolor='#F8FAFC', edgecolor='#CBD5E1',
                               linewidth=1.2, linestyle='-', antialiased=True))
    ax.add_patch(plt.Rectangle((65, 92), 7, 138, facecolor=ITB_BLUE, edgecolor='none'))

    ax.text(90, 200, 'LECTURE CREDITS & ACKNOWLEDGEMENT', fontsize=11.5, fontweight='bold',
            color=ITB_BLUE, fontfamily='sans-serif')
    ax.text(90, 172, 'Original course materials, illustrations, and formulations developed by:',
            fontsize=12, color='#334155', fontfamily='sans-serif')
    ax.text(90, 144, 'Prof. Jiajun Wu',
            fontsize=14, fontweight='bold', color='#0F172A', fontfamily='sans-serif')
    ax.text(90, 118, 'Stanford University — CS231n 10th Anniversary (Spring 2025, Lecture 15: 3D Vision)',
            fontsize=11.5, fontstyle='italic', color='#64748B', fontfamily='sans-serif')

    return make_page_from_fig(fig)


def create_credits_slide(slide_num, total_slides):
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_PT)
    ax.set_ylim(0, HEIGHT_PT)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 0), WIDTH_PT, HEIGHT_PT, facecolor='#FFFFFF', edgecolor='none'))
    draw_itb_chrome(ax, slide_num, total_slides)

    # Title
    ax.text(65, 460, 'Course Credits & Acknowledgements', fontsize=30, fontweight='bold',
            color=ITB_BLUE, fontfamily='sans-serif')
    ax.add_patch(plt.Rectangle((65, 442), 830, 1.2, facecolor='#E2E8F0', edgecolor='none'))

    # Context paragraph
    ax.text(65, 385,
            'This lecture presentation is adapted for the IF5152 Computer Vision course at Institut Teknologi Bandung (ITB).\n'
            'The technical content, structure, and visual materials originate from Stanford University\'s CS231n\n'
            '(Deep Learning for Computer Vision, 10th Anniversary Edition, Spring 2025).',
            fontsize=13.5, color='#334155', fontfamily='sans-serif', linespacing=1.5)

    # Authors Box
    ax.add_patch(plt.Rectangle((65, 115), 830, 235, facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2))
    ax.add_patch(plt.Rectangle((65, 115), 7, 235, facecolor=ITB_BLUE, edgecolor='none'))

    ax.text(90, 318, 'ORIGINAL LECTURE AUTHOR & CONTRIBUTORS', fontsize=12,
            fontweight='bold', color=ITB_BLUE, fontfamily='sans-serif')

    entries = [
        ('Prof. Jiajun Wu', 'Assistant Professor of Computer Science, Stanford University (Lecture 15 Author)'),
        ('Stanford CS231n Team', 'CS231n 10th Anniversary Curriculum (Spring 2025, May 22, 2025)'),
        ('Contributing Slide Authors', 'Prof. Ren Ng (UC Berkeley), Prof. Hao Su (UC San Diego), Prof. Daniel Ritchie (Brown Univ.)'),
        ('Research Citations', 'All cited works, figures, and benchmarks remain credited to their respective authors')
    ]

    y_pos = 278
    for label, desc in entries:
        ax.text(95, y_pos, f'•  {label}:', fontsize=13, fontweight='bold', color='#0F172A', fontfamily='sans-serif')
        ax.text(335, y_pos, desc, fontsize=11.5, color='#475569', fontfamily='sans-serif')
        y_pos -= 44

    return make_page_from_fig(fig)


def create_closing_slide(slide_num, total_slides):
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_PT)
    ax.set_ylim(0, HEIGHT_PT)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 0), WIDTH_PT, HEIGHT_PT, facecolor='#FFFFFF', edgecolor='none'))
    draw_itb_chrome(ax, slide_num, total_slides)

    ax.text(65, 455, '3D Vision: Key Takeaways', fontsize=30, fontweight='bold',
            color=ITB_BLUE, fontfamily='sans-serif')
    ax.add_patch(plt.Rectangle((65, 436), 830, 1.2, facecolor='#E2E8F0', edgecolor='none'))

    takeaways = [
        ('Explicit 3D Representations:', 'Voxels (3D CNNs), Point Clouds (PointNet / Chamfer distance), and Polygonal Meshes (mesh deformation, AtlasNet).'),
        ('Implicit 3D Representations:', 'Signed Distance Functions (DeepSDF) and Occupancy Networks enable continuous, resolution-independent geometry.'),
        ('Neural Radiance Fields (NeRF):', 'Volumetric ray-marching with coordinate MLPs and positional encoding for photorealistic novel view synthesis.'),
        ('3D Gaussian Splatting:', 'Explicit anisotropic 3D Gaussians with differentiable rasterization for real-time high-fidelity rendering.'),
        ('Structure-Aware 3D Models:', 'Part sets, relationship graphs, hierarchies (StructureNet), and shape programs capture compositional geometry.')
    ]

    y_pos = 385
    for title, desc in takeaways:
        ax.text(70, y_pos, f'•  {title}', fontsize=13.5, fontweight='bold', color='#0F172A', fontfamily='sans-serif')
        ax.text(95, y_pos - 23, desc, fontsize=12, color='#475569', fontfamily='sans-serif')
        y_pos -= 62

    return make_page_from_fig(fig)


def create_itb_canvas(slide_num, total_slides):
    """Creates a blank white 960x540 canvas with ITB top accent and bottom footer."""
    fig = plt.figure(figsize=(WIDTH_PT / 72, HEIGHT_PT / 72), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_PT)
    ax.set_ylim(0, HEIGHT_PT)
    ax.axis('off')

    ax.add_patch(plt.Rectangle((0, 0), WIDTH_PT, HEIGHT_PT, facecolor='#FFFFFF', edgecolor='none'))
    draw_itb_chrome(ax, slide_num, total_slides)

    return make_page_from_fig(fig)


def main():
    input_path = 'IF5152/3d-vision/3d-vision_2025.pdf'
    output_path = 'IF5152/3d-vision/IF5152_3D_Vision.pdf'

    reader = pypdf.PdfReader(input_path)
    num_orig = len(reader.pages)
    print(f'Original pages in {input_path}: {num_orig}')

    # Slide mapping:
    # 1: ITB Title Slide (replaces Page 1)
    # 2: Credits & Acknowledgements Slide (new)
    # 3..106: Original Pages 2..105 (0-indexed 1..104) -> 104 slides
    # 107: ITB Key Takeaways / Closing Slide (new)
    total_slides = 1 + 1 + (num_orig - 1) + 1
    print(f'Target total slides: {total_slides}')

    writer = pypdf.PdfWriter()
    transform = Transformation().scale(SCALE, SCALE).translate(TX, TY)

    # 1. Title Slide
    print('Adding Slide 1: ITB Title Slide')
    writer.add_page(create_title_slide(total_slides))

    # 2. Credits Slide
    print('Adding Slide 2: Credits & Acknowledgements Slide')
    writer.add_page(create_credits_slide(2, total_slides))

    # 3..106: Content Slides (original indices 1..num_orig-1)
    print(f'Processing {num_orig - 1} content slides...')
    for idx in range(1, num_orig):
        slide_num = idx + 2  # idx=1 -> Slide 3
        canvas = create_itb_canvas(slide_num, total_slides)
        src_page = reader.pages[idx]
        canvas.merge_transformed_page(src_page, transform)
        writer.add_page(canvas)

    # 107: Closing Slide
    print(f'Adding Slide {total_slides}: Closing Slide')
    writer.add_page(create_closing_slide(total_slides, total_slides))

    print(f'Writing output to {output_path}...')
    with open(output_path, 'wb') as f:
        writer.write(f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Done! Generated {output_path} ({total_slides} pages, {size_mb:.2f} MB)')


if __name__ == '__main__':
    main()

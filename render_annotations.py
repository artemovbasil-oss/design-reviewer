
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

def _gradient_bg(size: Tuple[int,int]) -> Image.Image:
    w,h = size
    img = Image.new("RGBA", (w,h), (0,0,0,255))
    top = random.randint(140, 200)
    bot = random.randint(10, 30)
    for y in range(h):
        t = y/(h-1)
        b = int((1-t)*top + t*bot)
        r = int((1-t)*10 + t*0)
        g = int((1-t)*30 + t*0)
        img.paste((r,g,b,255), (0,y,w,y+1))
    return img

def _load_font(pref: str, size: int):
    try:
        return ImageFont.truetype(pref, size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

def _shadow(image: Image.Image, radius: int = 36, offset=(0,8), alpha=100):
    w,h = image.size
    shadow = Image.new("RGBA", (w+abs(offset[0])*2, h+abs(offset[1])*2), (0,0,0,0))
    sh = Image.new("RGBA", (w, h), (0,0,0,alpha))
    sh = sh.filter(ImageFilter.GaussianBlur(radius))
    shadow.paste(sh, (abs(offset[0]), abs(offset[1])), sh)
    return shadow

def _draw_rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def _fit_inside(img: Image.Image, target_wh):
    W,H = target_wh
    w,h = img.size
    k = min(W/w, H/h)
    new = (max(1,int(w*k)), max(1,int(h*k)))
    return img.resize(new, Image.LANCZOS)

def render_annotations(
    screenshot: Image.Image,
    lines: List[Dict[str, Any]],
    findings: List[Dict[str, Any]],
    user_name: Optional[str] = None,
    logo_path: Optional[str] = None,
    canvas_size: Tuple[int,int] = (1280,1280)
) -> Image.Image:
    W,H = canvas_size
    bg = _gradient_bg((W,H)).convert("RGBA")
    canvas = Image.new("RGBA", (W,H), (0,0,0,0))
    canvas.alpha_composite(bg)

    font_title = _load_font("Inter-SemiBold.ttf", 48)
    font_text  = _load_font("Inter-Regular.ttf", 32)
    font_badge = _load_font("Inter-SemiBold.ttf", 36)
    font_small = _load_font("Inter-Regular.ttf", 26)

    max_w = int(W*0.86)
    max_h = int(H*0.76)
    shot = _fit_inside(screenshot, (max_w, max_h)).convert("RGBA")

    shadow = _shadow(shot, radius=48, offset=(0,18), alpha=130)
    sx = (W - shadow.size[0])//2
    sy = (H - shadow.size[1])//2 - 24
    canvas.alpha_composite(shadow, (sx, sy))

    cx = (W - shot.size[0])//2
    cy = (H - shot.size[1])//2 - 24
    card = Image.new("RGBA", shot.size, (255,255,255,255))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle([0,0,shot.size[0]-1,shot.size[1]-1], radius=28, outline=(230,235,238,255), width=2)
    card.alpha_composite(shot)
    canvas.alpha_composite(card, (cx, cy))

    d = ImageDraw.Draw(canvas)
    frame_color = (0, 200, 255, 220)
    ow, oh = screenshot.size
    scale = min(shot.size[0]/ow, shot.size[1]/oh)
    nw, nh = int(ow*scale), int(oh*scale)
    ox = cx + (shot.size[0]-nw)//2
    oy = cy + (shot.size[1]-nh)//2
    for f in findings or []:
        bbox = f.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1,y1,x2,y2 = bbox
        rx1 = ox + int(x1*scale); ry1 = oy + int(y1*scale)
        rx2 = ox + int(x2*scale); ry2 = oy + int(y2*scale)
        _draw_rounded_rect(d, (rx1,ry1,rx2,ry2), radius=16, outline=frame_color, width=3)

    has_issues = bool(findings)
    badge_text = "✅ Кайф" if not has_issues else "❌ Не кайф"
    badge_fill = (36, 142, 255, 255) if not has_issues else (206, 59, 74, 255)
    badge_w = 280; badge_h = 86; radius = 24
    bx = W - badge_w - 56
    by = H - badge_h - 56
    d.rounded_rectangle([bx,by,bx+badge_w,by+badge_h], radius=radius, fill=badge_fill)
    uname = user_name or "гость"
    tb = d.textbbox((0,0), uname, font=font_small)
    d.text((bx + badge_w - (tb[2]-tb[0]), by - (tb[3]-tb[1]) - 16), uname, font=font_small, fill=(230,235,238,255))
    tbb = d.textbbox((0,0), badge_text, font=font_badge)
    d.text((bx + (badge_w-(tbb[2]-tbb[0]))//2, by + (badge_h-(tbb[3]-tbb[1]))//2), badge_text, font=font_badge, fill=(255,255,255,255))

    lp = logo_path or "лого.png"
    try:
        logo = Image.open(lp).convert("RGBA")
        L = 132
        k = min(L/logo.width, L/logo.height)
        logo = logo.resize((max(1,int(logo.width*k)), max(1,int(logo.height*k))), Image.LANCZOS)
        lx, ly = 56, H - logo.height - 56
        d.rounded_rectangle([lx-16,ly-16,lx+logo.width+16,ly+logo.height+16], radius=20, fill=(255,255,255,30))
        canvas.alpha_composite(logo, (lx, ly))
    except Exception:
        d.text((56, H-56-28), "АвтоДушнила", font=font_text, fill=(230,235,238,255))

    return canvas.convert("RGB")

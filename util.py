
import cv2
import numpy as np

def creepy_bw_effect(bgr_img, strength=1.0):
    if strength <= 0:
        return bgr_img

    h, w = bgr_img.shape[:2]

    # 1) 轉灰階 + CLAHE 對比
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 2) unsharp mask
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    # 3) 顆粒
    noise = np.random.normal(0, 12, (h, w)).astype(np.float32)
    noisy = np.clip(sharp.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4) 暗角
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    r = np.sqrt(((xx - cx)**2 + (yy - cy)**2))
    vignette = 1.0 - np.clip(r / (0.8*max(w, h)), 0, 1)  # 中心亮、邊緣暗
    vignette = (vignette * (0.6 + 0.4*(1-strength)))  # 強度越大邊緣越暗
    v_img = np.clip(noisy.astype(np.float32) * vignette.astype(np.float32), 0, 255).astype(np.uint8)

    # 5) 輕微水平抖動（ghosting）
    shift = int(2 + 4*strength)  # 2~6 pixels
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    jitter = cv2.warpAffine(v_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # merge 為 BGR
    creepy_bgr = cv2.cvtColor(jitter, cv2.COLOR_GRAY2BGR)
    return creepy_bgr

def mix_effect(color_img, creepy_img, alpha):
    """以 alpha 進行線性混合，alpha=1 完全詭異，0 完全彩色"""
    alpha = np.clip(alpha, 0.0, 1.0)
    return cv2.addWeighted(creepy_img, alpha, color_img, 1.0 - alpha, 0)

def bbox_area_ratio(inf_result, frame_w, frame_h):
    """計算偵測結果中，最大 bbox 面積佔整張影像的比例"""
    max_ratio = 0.0
    if inf_result is None or len(inf_result) == 0:
        return max_ratio

    for det in inf_result:
        x1, y1, x2, y2 = det['bbox']
        x1 = int(np.clip(x1, 0, frame_w-1))
        y1 = int(np.clip(y1, 0, frame_h-1))
        x2 = int(np.clip(x2, 0, frame_w-1))
        y2 = int(np.clip(y2, 0, frame_h-1))
        area = (x2 - x1) * (y2 - y1)
        ratio = area / (frame_w * frame_h)
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

def extract_dets_for_ratio(inf_result):
    dets = []
    try:
        for r in inf_result.results:
            if 'bbox' in r:
                dets.append({'bbox': r['bbox']})
            elif 'box' in r:
                dets.append({'bbox': r['box']})
    except Exception:
        pass
    return dets

def compute_alpha_from_ratio(ratio, no_face):
    if no_face:
        return 1.0
    cap = 0.001
    if ratio >= cap:
        return 0.0
    return (cap - ratio) / cap

def weak_signal_flicker(bgr_img,
                        strength=1.0, 
                        state=None,
                        trigger_prob=0.01, 
                        min_len=3, 
                        max_len=10,
                        allowed_modes=None,
                        mode_weights=None,
    ):
    """
    模擬老電視訊號不穩的偶發閃爍/失真效果。
    - strength: 0~1，控制效果強度與啟動機率。
    - state: 跨幀狀態 dict；若為 None 會自動建立。
      會使用/產生的鍵：
        - remain: 本次效果還要持續幾幀（>0 代表效果啟動中）
        - total:  本次效果總幀數（用於入/出場漸變）
        - mode:   本次採用的故障模式
        - vy:     （僅 roll 模式）目前垂直偏移的累積量
    - trigger_prob: 每幀觸發基礎機率（會再乘上 strength）
    - min_len/max_len: 每次觸發後持續的幀數範圍（含）
    回傳： (out_img_uint8, state)
    """
    import numpy as _np
    import cv2 as _cv2

    if state is None:
        state = {"remain": 0}

    h, w = bgr_img.shape[:2]
    strength = float(_np.clip(strength, 0.0, 1.0))

    all_modes = ["noise", "desync", "contrast", "blackout", "whiteout", "roll"]
    default_p = _np.array([0.30, 0.20, 0.25, 0.08, 0.07, 0.10], dtype=_np.float32)

    modes = all_modes if allowed_modes is None else [m for m in all_modes if m in allowed_modes]
    if len(modes) == 0:
        modes = ["noise", "desync", "contrast", "roll"]  # fallback

    if mode_weights is not None:
        # 用自訂權重
        p = _np.array([float(mode_weights.get(m, 0.0)) for m in modes], dtype=_np.float32)
        if p.sum() <= 0:
            p = _np.ones(len(modes), dtype=_np.float32)
    else:
        # 從預設機率過濾
        idxs = [all_modes.index(m) for m in modes]
        p = default_p[idxs]

    p = p / p.sum()

    # 若目前沒有在閃爍，嘗試觸發一次
    if state.get("remain", 0) <= 0:
        if _np.random.rand() < (trigger_prob * (0.4 + 0.6 * strength)):
            dur = int(_np.random.randint(min_len, max_len + 1))
            mode = _np.random.choice(modes, p=p)  # ← 改用上面 modes/p
            state.update({"remain": dur, "total": dur, "mode": str(mode)})
            if mode == "roll" and "vy" not in state:
                state["vy"] = 0
        else:
            return bgr_img, state

    # 進入效果期
    remain = int(state.get("remain", 1))
    total = max(int(state.get("total", remain)), 1)
    mode = state.get("mode", "noise")

    # 以餘數做入/出場權重（邊緣較弱，中段較強）
    t = 1.0 - (remain - 1) / total
    env = (1.0 - _np.cos(_np.pi * t)) * 0.5  # 0→1 平滑
    power = strength * (0.5 + 0.5 * env)     # 讓中段最強

    out = bgr_img.astype(_np.float32).copy()

    if mode == "noise":
        # 高斯雜訊 + salt & pepper + 偶發水平亮線
        sigma = 10 + 45 * power
        gauss = _np.random.normal(0, sigma, bgr_img.shape).astype(_np.float32)
        out = _np.clip(out + gauss, 0, 255)

        sp_p = 0.002 + 0.02 * power
        mask = _np.random.rand(h, w)
        out[mask < sp_p] = 0
        out[mask > 1 - sp_p] = 255

        # 幾條隨機亮線
        for _ in range(int(1 + 5 * power)):
            y = int(_np.random.randint(0, h))
            thickness = int(1 + 2 * power)
            _cv2.line(out, (0, y), (w - 1, y), (255, 255, 255), thickness)

    elif mode == "desync":
        # 每一行有不同的水平位移（水平不同步/撕裂）
        max_shift = int(4 + 18 * power)  # 像素
        shifts = (_np.random.randn(h) * max_shift * 0.4).astype(_np.int32)
        for y in range(h):
            s = int(_np.clip(shifts[y], -max_shift, max_shift))
            if s == 0:
                continue
            row = out[y]
            out[y] = _np.roll(row, s, axis=0)
        # 在一兩個區段加強撕裂
        for _ in range(int(1 + 2 * power)):
            y0 = int(_np.random.randint(0, max(1, h - 10)))
            y1 = int(_np.clip(y0 + _np.random.randint(3, 20), 0, h))
            out[y0:y1] = _np.clip(out[y0:y1] * (0.8 - 0.3 * power), 0, 255)

    elif mode == "contrast":
        # 整體亮度/對比快速抖動 + 掃描線
        gain = 0.6 + 0.6 * (1.0 - power)  # 越強越黯
        out = _np.clip(out * gain, 0, 255)
        # gamma 隨機化一點點
        gamma = 1.0 + (_np.random.randn() * 0.25 * power)
        gamma = _np.clip(gamma, 0.6, 1.6)
        out = _np.clip(_np.power(out / 255.0, gamma) * 255.0, 0, 255)

        # 掃描線（隔行變暗）
        scan = _np.tile((_np.arange(h) % 2).reshape(-1, 1, 1), (1, w, 3))
        out[scan == 1] *= (0.85 - 0.25 * power)

    elif mode == "blackout":
        # 黑場但帶一點噪點與行同步紋
        level = 25 * (0.3 + 0.7 * (1 - power))
        out[:] = level
        noise = _np.random.normal(0, 25 + 60 * power, bgr_img.shape)
        out = _np.clip(out + noise, 0, 255)
        for _ in range(int(2 + 8 * power)):
            y = int(_np.random.randint(0, h))
            _cv2.line(out, (0, y), (w - 1, y), (level + 50, level + 50, level + 50), 1)

    elif mode == "whiteout":
        # 白場 + 些微波動與雜點
        level = 255 - 20 * (0.3 + 0.7 * (1 - power))
        out[:] = level
        noise = _np.random.normal(0, 15 + 30 * power, bgr_img.shape)
        out = _np.clip(out + noise, 0, 255)

    elif mode == "roll":
        # 垂直滾動（畫面向上/下慢慢滾），加一條 tearing 線
        vy = int(state.get("vy", 0))
        step = int(2 + 10 * power)
        direction = -1 if _np.random.rand() < 0.5 else 1
        vy = (vy + direction * step) % h
        out = _np.roll(out, vy, axis=0)
        state["vy"] = vy

        # 撕裂線：在某條水平線做左右錯位
        tear_y = int((vy + h // 3) % h)
        tear_shift = int(8 + 22 * power)
        left = out[:tear_y]
        right = out[tear_y:]
        if left.size and right.size:
            right = _np.roll(right, tear_shift, axis=1)
            out = _np.vstack([left, right])
        # 降一點對比讓感覺更像掃描
        out *= (0.9 - 0.2 * power)

    # 轉回 uint8
    out = _np.clip(out, 0, 255).astype(_np.uint8)

    # 更新剩餘幀數
    state["remain"] = remain - 1
    if state["remain"] <= 0:
        # 清除一次性狀態
        state["remain"] = 0
        state.pop("total", None)
        state.pop("mode", None)
        state.pop("vy", None)

    return out, state

def render_flicker_sprite(
    bgr_img,
    sprite_rgba,
    state=None,
    trigger_prob=0.01,
    min_len=8,
    max_len=28,
    base_scale=0.75,
    jitter_px=6,
):
    """
    在畫面上隨機觸發一個帶抖動/閃爍/淡入淡出的 sprite（支援 PNG 透明）。
    - sprite_rgba: 可為單一 HxWx4 影像，或影像清單(list/tuple)
    - state: dict 跨幀狀態（含 remain/total/x/y/scale/spr_idx 等）
    回傳：(out_bgr, state)
    """
    import numpy as _np
    import cv2 as _cv2

    # 允許傳入 list/tuple；過濾掉 None 或空影像
    if isinstance(sprite_rgba, (list, tuple)):
        sprites = [s for s in sprite_rgba if s is not None and getattr(s, "size", 0) > 0]
    else:
        sprites = [sprite_rgba] if (sprite_rgba is not None and getattr(sprite_rgba, "size", 0) > 0) else []

    if not sprites:
        return bgr_img, {"remain": 0}

    h, w = bgr_img.shape[:2]
    if state is None:
        state = {"remain": 0}

    # 是否需要在這一幀觸發
    if state.get("remain", 0) <= 0:
        if _np.random.rand() < trigger_prob:
            dur = int(_np.random.randint(min_len, max_len + 1))

            # 這次出現要用哪一張？此刻隨機決定，整段期間固定
            spr_idx = int(_np.random.randint(0, len(sprites)))
            spr0 = sprites[spr_idx]
            sh, sw = spr0.shape[:2]

            # 隨機大小與位置（置中偏下）
            scale = base_scale * (0.9 + 0.3 * _np.random.rand())
            tw, th = int(sw * scale), int(sh * scale)
            cx = int(w * (0.5 + 0.06 * (_np.random.rand() - 0.5)))
            cy = int(h * (0.60 + 0.06 * (_np.random.rand() - 0.5)))
            x = int(cx - tw // 2)
            y = int(cy - th // 2)

            state.update({
                "remain": dur,
                "total": dur,
                "x": x, "y": y,
                "scale": float(scale),
                "phase": float(_np.random.rand() * 2 * _np.pi),
                "spr_idx": spr_idx,  # 記住這次用哪一張
            })
        else:
            return bgr_img, state

    # 讀 state
    remain = int(state.get("remain", 1))
    total  = max(int(state.get("total", remain)), 1)
    x, y   = int(state.get("x", 0)), int(state.get("y", 0))
    scale  = float(state.get("scale", base_scale))
    phase  = float(state.get("phase", 0.0))
    spr_idx = int(state.get("spr_idx", 0)) % len(sprites)
    spr = sprites[spr_idx]

    # 淡入淡出曲線 + 微閃爍
    t = 1.0 - (remain - 1) / total           # 0→1
    fade = (1.0 - _np.cos(_np.pi * t)) * 0.5 # 0→1→0
    flicker = 0.75 + 0.25 * _np.sin(10*t + phase)
    alpha_mul = _np.clip(fade * flicker, 0.0, 1.0)

    # 抖動
    jx = int((_np.random.randn()) * jitter_px)
    jy = int((_np.random.randn()) * jitter_px)

    # 拆出 RGB 與 alpha
    if spr.shape[2] == 4:
        spr_rgb = spr[..., :3]
        spr_a   = spr[..., 3:4] / 255.0
    else:
        spr_rgb = spr
        spr_a   = _np.ones((*spr.shape[:2], 1), dtype=_np.float32)

    # 縮放
    sh, sw = spr_rgb.shape[:2]
    tw, th = max(1, int(sw * scale)), max(1, int(sh * scale))
    spr_rgb = _cv2.resize(spr_rgb, (tw, th), interpolation=_cv2.INTER_LINEAR)
    spr_a   = _cv2.resize(spr_a,   (tw, th), interpolation=_cv2.INTER_LINEAR)
    if spr_a.ndim == 2:
        spr_a = spr_a[:, :, None]

    # ROI 放置
    xx = _np.clip(x + jx, 0, w-1)
    yy = _np.clip(y + jy, 0, h-1)
    x2 = int(_np.clip(xx + tw, 0, w))
    y2 = int(_np.clip(yy + th, 0, h))
    tw2 = x2 - int(xx)
    th2 = y2 - int(yy)

    if tw2 <= 1 or th2 <= 1:
        out = bgr_img.copy()
    else:
        spr_rgb = spr_rgb[:th2, :tw2]
        spr_a   = spr_a[:th2, :tw2] * alpha_mul

        roi = bgr_img[int(yy):y2, int(xx):x2].astype(_np.float32)
        spr_rgb_f = spr_rgb.astype(_np.float32)
        a = spr_a.astype(_np.float32)

        blended = spr_rgb_f * a + roi * (1.0 - a)
        out = bgr_img.copy()
        out[int(yy):y2, int(xx):x2] = _np.clip(blended, 0, 255).astype(_np.uint8)

    # 更新狀態；結束時清掉 spr_idx
    state["remain"] = remain - 1
    if state["remain"] <= 0:
        state.clear()
        state["remain"] = 0

    return out, state

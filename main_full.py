import os
import cv2
import numpy as np
import degirum as dg
# import degirum_tools  # ← 不再使用 OpenCV 以外的顯示

from config import (
    INFERENCE_HOST,
    ZOO_URL,
    TOKEN,
    DEVICE_TYPE,
    MODEL_NAME,
    CAM_INDEX,
    FRAME_W,
    FRAME_H,
    FULLSCREEN,
    SCREEN_MODE,
    SCREEN_W,
    SCREEN_H,
)

from util import creepy_bw_effect, mix_effect, extract_dets_for_ratio, weak_signal_flicker, render_flicker_sprite

CONF_TH = 0.35
DRAW_BOX = False
SHOW_DEBUG_TEXT = False
MIRROR = True
FLICKER_ENABLED = True
GHOST_ENABLED = True
GHOST_TRIGGER_P = 0.08
WINDOW_NAME = "Hello Ghost"

def draw_overlay_text(overlay, face_cnt, flicker_on):
    if not SHOW_DEBUG_TEXT:
        return
    status = "FLICKER" if flicker_on else "NORMAL"
    cv2.putText(
        overlay,
        f"Faces: {face_cnt}  Mode: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

def _get_screen_resolution(mode, sw, sh):
    # 針對你的兩種螢幕給預設值；其餘嘗試自動偵測
    if mode == "4k":
        return 3840, 2160
    if mode == "7inch":
        return 1024, 600
    if mode == "custom" and sw > 0 and sh > 0:
        return int(sw), int(sh)
    # auto：用 tkinter 偵測，失敗退 1920x1080 或你填的 sw/sh
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        if sw > 0 and sh > 0:
            return int(sw), int(sh)
        return 1920, 1080

def main():
    # 攝影機初始化（維持原本解析度）
    cam = cv2.VideoCapture(CAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cam.isOpened():
        print("無法開啟攝影機")
        return

    # 取得顯示解析度
    SCREEN_W_ACT, SCREEN_H_ACT = _get_screen_resolution(SCREEN_MODE, SCREEN_W, SCREEN_H)

    # 建立 OpenCV 視窗並切到全螢幕（無邊框）
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        # 這會讓視窗無邊框 + 充滿螢幕
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # 非全螢幕時，仍可把視窗設到指定大小
        cv2.resizeWindow(WINDOW_NAME, SCREEN_W_ACT, SCREEN_H_ACT)

    flicker_state = {"remain": 0}
    ghost_state = {"remain": 0}

    model = dg.load_model(
        model_name=MODEL_NAME,
        inference_host_address=INFERENCE_HOST,
        zoo_url=ZOO_URL,
        token=TOKEN,
        device_type=DEVICE_TYPE,
    )

    # 載入鬼 PNG
    ghost_imgs = []
    for name in ("ghost_1.png", "ghost_2.png"):
        img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if img is not None:
            ghost_imgs.append(img)

    if not ghost_imgs:
        print("No ghost images found, disabling ghost effect.")
        GHOST_LOCAL = False
    else:
        GHOST_LOCAL = True

    while True:
        ok, frame = cam.read()
        if not ok:
            print("無法讀取畫面")
            break

        if MIRROR:
            frame = cv2.flip(frame, 1)

        # --- 推論 ---
        inf = model(frame)

        # 判定是否有臉
        dets = extract_dets_for_ratio(inf)
        face_cnt = len(dets)
        face_present = face_cnt > 0

        # --- 視覺邏輯 ---
        if face_present:
            overlay = frame
            flicker_on = False
            flicker_state["remain"] = 0
            for k in ("total", "mode", "vy"):
                flicker_state.pop(k, None)
        else:
            creepy = creepy_bw_effect(frame, strength=1.0)
            base_scene = mix_effect(frame, creepy, alpha=1.0)

            scene_with_ghost = base_scene
            ghost_active = False
            if GHOST_ENABLED and GHOST_LOCAL:
                scene_with_ghost, ghost_state = render_flicker_sprite(
                    base_scene,
                    ghost_imgs,
                    state=ghost_state,
                    trigger_prob=GHOST_TRIGGER_P,
                    min_len=8,
                    max_len=28,
                    base_scale=0.75,
                    jitter_px=3,
                )
                ghost_active = ghost_state.get("remain", 0) > 0

            flicker_on = FLICKER_ENABLED
            if FLICKER_ENABLED:
                allowed = ["noise", "desync"] if ghost_active else None
                overlay, flicker_state = weak_signal_flicker(
                    scene_with_ghost,
                    strength=1.0,
                    state=flicker_state,
                    trigger_prob=0.01,
                    min_len=3,
                    max_len=5,
                    allowed_modes=allowed,
                )
            else:
                overlay = scene_with_ghost

        draw_overlay_text(overlay, face_cnt, flicker_on)

        # ★ 依螢幕真實解析度輸出（避免縮放由視窗/WM 隨機處理）
        if overlay.shape[1] != SCREEN_W_ACT or overlay.shape[0] != SCREEN_H_ACT:
            out = cv2.resize(overlay, (SCREEN_W_ACT, SCREEN_H_ACT), interpolation=cv2.INTER_LINEAR)
        else:
            out = overlay

        cv2.imshow(WINDOW_NAME, out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

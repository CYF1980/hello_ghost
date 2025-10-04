import os 
import cv2
import numpy as np
import degirum as dg
import degirum_tools

from config import (
    INFERENCE_HOST,
    ZOO_URL,
    TOKEN,
    DEVICE_TYPE,
    MODEL_NAME,
    CAM_INDEX,
    FRAME_W,
    FRAME_H,
)

from util import creepy_bw_effect, mix_effect, extract_dets_for_ratio, weak_signal_flicker, render_flicker_sprite

CONF_TH = 0.35          # Confidence threshold for drawing box
DRAW_BOX = False        # Draw face bounding box or not
SHOW_DEBUG_TEXT = False  # Show debug text on screen or not
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

def main():
    cam = cv2.VideoCapture(CAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cam.isOpened():
        print("無法開啟攝影機")
        return
     
    flicker_state = {"remain": 0}
    model = dg.load_model(
        model_name=MODEL_NAME,
        inference_host_address=INFERENCE_HOST,
        zoo_url=ZOO_URL,
        token=TOKEN,
        device_type=DEVICE_TYPE,
    )

    ghost_state = {"remain": 0}
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

    # 顯示器（DeGirum 提供的便利視窗）
    with degirum_tools.Display(WINDOW_NAME, show_fps=False) as display:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("無法讀取畫面")
                break

            if MIRROR:
                frame = cv2.flip(frame, 1)

            # --- 推論 ---
            inf = model(frame)

            # 取偵測結果（只要有一張臉就算偵測到）
            dets = extract_dets_for_ratio(inf)  # 重用 util 裡的解析函式
            face_cnt = len(dets)
            face_present = face_cnt > 0

            # --- 視覺邏輯 ---
            if face_present:
                # 有臉：回到正常畫面；並把閃爍狀態清空
                overlay = frame
                flicker_on = False
                flicker_state["remain"] = 0
                # 清除一次性狀態，避免殘影
                for k in ("total", "mode", "vy"):
                    flicker_state.pop(k, None)
            else:
                creepy = creepy_bw_effect(frame, strength=1.0)
                base_scene = mix_effect(frame, creepy, alpha=1.0)  # 完全詭異
            
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
                # 無臉：啟動/持續弱訊號閃爍效果
                flicker_on = FLICKER_ENABLED
                if FLICKER_ENABLED:
                    # all:     "noise", "desync", "contrast", "blackout", "whiteout", "roll"
                    allowed = ["noise", "desync"] if ghost_active else None
                    overlay, flicker_state = weak_signal_flicker(
                        scene_with_ghost,
                        strength=1.0,
                        state=flicker_state,
                        trigger_prob=0.01,
                        min_len=3,
                        max_len=5,
                        allowed_modes=allowed,   # ← 新參數
                        # 也可加上 mode_weights 調口味，例如：{"noise":3,"desync":2,"contrast":2,"roll":1}
                    )
                else:
                    overlay = scene_with_ghost 

            # 選配：在畫面上畫出人臉框（僅供除錯或示意）
            if DRAW_BOX and face_present:
                try:
                    for r in inf.results:
                        conf = r.get('confidence', 1.0)
                        if conf < CONF_TH:
                            continue
                        bb = r.get('bbox', r.get('box', None))
                        if bb is None or len(bb) != 4:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bb]
                        x1 = max(0, min(x1, FRAME_W - 1))
                        y1 = max(0, min(y1, FRAME_H - 1))
                        x2 = max(0, min(x2, FRAME_W - 1))
                        y2 = max(0, min(y2, FRAME_H - 1))
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception:
                    pass

            draw_overlay_text(overlay, face_cnt, flicker_on)

            display.show(overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

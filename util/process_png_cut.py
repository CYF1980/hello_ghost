from PIL import Image

# 開啟圖片
img = Image.open("ghost_2.png")
width, height = img.size

crop_box = (0, 0, width, height*2 // 3)
cropped = img.crop(crop_box)

# 存檔
cropped.save("output_top_half.png")

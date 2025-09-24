from PIL import Image

# 開啟圖片
img = Image.open("output_top_half1.png")

resized = img.resize((480, 480))

# 存檔
resized.save("output.png")

# import cv2
# import numpy as np

# # 读取图像
# image = cv2.imread('input.png')
# height, width = image.shape[:2]

# # 转换为灰度图并进行边缘检测
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150)

# # 霍夫线变换检测所有直线
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# # 筛选中间区域的垂直线
# vertical_lines = []
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
#         # 筛选垂直直线且在中间1/3区域
#         if 80 <= angle <= 100 and (width/3 < x1 < 2*width/3):
#             vertical_lines.append((x1, x2))

# # 确定最终竖线位置（取x坐标平均值）
# if vertical_lines:
#     x_values = [x for line in vertical_lines for x in line]
#     x_center = int(np.mean(x_values))
#     # 绘制红框（宽度10像素）
#     cv2.rectangle(image, (x_center-5, 0), (x_center+5, height), (0, 0, 255), 2)
#     cv2.imwrite('output.png', image)
#     print("处理完成，已保存为output.png")
# else:
#     print("未检测到中间竖线")


import cv2
import numpy as np

# 读取图像
def find_center_line(image_path):
    image_path = str(image_path)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return [None, "无法读取图片"]
    height, width = image.shape[:2]

    # 转换为灰度图并进行边缘检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 霍夫线变换检测所有直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # 筛选中间区域的垂直线
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            # 筛选垂直直线且在中间1/3区域
            if 80 <= angle <= 100 and (width/3 < x1 < 2*width/3):
                vertical_lines.append((x1, x2))

    ret=[]
    # 确定最终竖线位置（取x坐标平均值）
    if vertical_lines:
        x_values = [x for line in vertical_lines for x in line]
        x_center = int(np.mean(x_values))
        # 绘制红框（宽度10像素）
        cv2.rectangle(image, (x_center-5, 0), (x_center+5, height), (0, 0, 255), 2)
        cv2.imwrite('output.png', image)
        print("处理完成，已保存为output.png")
        ret.append(x_center)
        ret.append("处理完成，已保存为output.png")
        return ret
    else:
        print("未检测到中间竖线")
        ret.append(None)
        ret.append("未检测到中间竖线")
        return ret

# 示例用法

if __name__ == '__main__':
    image_path = 'input.png'  # 替换为你的图片路径
    image_path = r"D:\space\python\aichain\shuati\output_高中必刷题数学人教A版必修1/page_1.png"
# find_center_line(image_path)
    find_center_line(image_path)

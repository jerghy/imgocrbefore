import fitz  # PyMuPDF
import os
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
name="高中必刷题数学人教A版必修1.pdf"
# url = "https://chogo.teracloud.jp/dav/documents/output.mp3"
url = "https://chogo.teracloud.jp/dav/shuati/"+name
auth = HTTPBasicAuth("ThomasXie", "43rKo29cev5Uzbyp")

response = requests.get(url, auth=auth, stream=True)

if response.status_code == 200:
    total_size = int(response.headers.get('content-length', 0))
    with open(name, "wb") as f:
        for data in tqdm(response.iter_content(1024), total=total_size // 1024, unit='KB'):
            f.write(data)
    print("下载成功")
else:
    print(f"下载失败，状态码：{response.status_code}")


def pdf_to_images(pdf_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    
    # 遍历每一页并保存为图片
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()  # 降低分辨率
        pix.save(f'{output_folder}/page_{page_num+1}.png')  # 最大压缩

# 使用示例
pdf_name = '高中必刷题数学人教A版必修1.pdf'
current_file_path = os.path.abspath(__file__)
pdf_path = os.path.join(os.path.dirname(current_file_path), pdf_name)
print(pdf_path)
output_folder = f'output_{pdf_name.split(".")[0]}'

pdf_to_images(pdf_path, output_folder)


import time
from findline import find_center_line
import fitz  # PyMuPDF
import json
import os
import cv2
import numpy as np
from PIL import Image
pdf_name = '高中必刷题数学人教A版必修1.pdf'
current_file_path = os.path.abspath(__file__)
pdf_path = os.path.join(os.path.dirname(current_file_path), pdf_name)
print(pdf_path)
output_folder = f'output_{pdf_name.split(".")[0]}'

pdf_document = fitz.open(pdf_path)

def split_image(image_path, x_center):
        
        
        # 读取图片
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None, None
        
        height, width = image.shape[:2]
        
        # 分割图片
        left_image = image[:, :x_center]
        right_image = image[:, x_center:]
        
        return left_image, right_image
images=[]
# 遍历每一页range(len(pdf_document))
results={}
pdfnum=0

for page_num in range(12,len(pdf_document)-12):
    pg=f'{output_folder}/page_{page_num+1}.png'
    pg=os.path.join(os.path.dirname(current_file_path),pg)
    print(pg)
    rets=find_center_line(pg)
    print(rets)
    time.sleep(0.1)
    if rets[0] != None:
        x_center = rets[0]
        left_image, right_image=split_image(pg, x_center)
        images.append(left_image)
        images.append(right_image)
        results[f"{page_num+1}.png"] = [pdfnum,pdfnum+1]
        pdfnum+=2
    else:
         image = cv2.imdecode(np.fromfile(pg, dtype=np.uint8), cv2.IMREAD_COLOR)
         images.append(image)
         results[f"{page_num+1}.png"] = [pdfnum]
         pdfnum+=1


with open("pdf.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
#把images保存为pdf
# 把images保存为pdf
output_pdf_path = os.path.join(os.path.dirname(current_file_path), 'output_split.pdf')

# 将OpenCV格式的图像转换为PIL格式
pil_images = []
for img in images:
    # OpenCV使用BGR，需要转换为RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pil_images.append(pil_img)

# 保存为压缩PDF
pil_images[0].save(
    output_pdf_path,
    "PDF",
    resolution=150.0,
    save_all=True,
    append_images=pil_images[1:],
    optimize=True,
    quality=60
)

print(f"PDF已保存到: {output_pdf_path}")



from imgocr import ImgOcr
# import cv2
import fitz  # PyMuPDF
import json
import os
# 初始化OCR模型
ocr = ImgOcr()
outputrelease={}

pdf_name = 'output_split.pdf'
current_file_path = os.path.abspath(__file__)
pdf_path = os.path.join(os.path.dirname(current_file_path), pdf_name)
print(pdf_path)
output_folder = f'output_{pdf_name.split(".")[0]}'

pdf_document = fitz.open(pdf_path)
# 遍历每一页range(len(pdf_document))
for page_num in range(len(pdf_document)):
    pg=f'{output_folder}/page_{page_num+1}.png'
    result = ocr.ocr(pg)
    print(pg)
    outputrelease[pg]=result





# 读取图片并进行OCR识别

# 保存result到JSON文件
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(outputrelease, f, ensure_ascii=False, indent=4)




from fastapi import FastAPI, HTTPException
import ddddocr
import uvicorn
from typing import Dict
import base64
from pydantic import BaseModel
import os

app = FastAPI(title="OCR API")

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化 ddddocr
try:
    ocr = ddddocr.DdddOcr(
        det=False,
        ocr=False,
        show_ad=False,
        import_onnx_path=os.path.join(current_dir, "models/95%.onnx"),
        charsets_path=os.path.join(current_dir, "models/charsets.json")
    )
    print("OCR initialized successfully")
except Exception as e:
    print(f"OCR initialization error: {str(e)}")
    raise

# 定义请求模型
class ImageRequest(BaseModel):
    image: str

@app.post("/ocr/", response_model=Dict[str, str])
async def read_image(request: ImageRequest):
    """
    接收base64编码的图片字符串进行OCR识别
    """
    try:
        # 处理可能包含的base64前缀
        image = request.image
        if "base64," in image:
            # 如果包含前缀，去除前缀
            image = image.split("base64,")[1]
        
        # 解码base64图片
        try:
            image_bytes = base64.b64decode(image)
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid base64 string")
            
        # 进行OCR识别
        try:
            result = ocr.classification(image_bytes)
            print(f"OCR result: {result}")
            return {"result": result}
        except Exception as e:
            print(f"OCR error: {str(e)}")
            raise HTTPException(status_code=500, detail="OCR processing failed")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    测试接口是否正常运行
    """
    return {"message": "OCR service is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)

# 读取图片并转换为base64
with open("path/to/your/image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

url = "http://localhost:80/ocr/"
data = {"image": base64_image}
response = requests.post(url, json=data)
print(response.json())
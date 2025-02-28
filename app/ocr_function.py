import ddddocr
import base64
from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel
import uvicorn
import os
from urllib.parse import unquote

# 初始化 FastAPI 应用
app = FastAPI(title="OCR API")

# 初始化 ddddocr
try:
    ocr = ddddocr.DdddOcr(
        det=False,
        ocr=False,
        show_ad=False,
        import_onnx_path="models/95%.onnx",  # 修改为相对路径
        charsets_path="models/charsets.json"  # 修改为相对路径
    )
    print("OCR initialized successfully")
except Exception as e:
    print(f"OCR initialization error: {str(e)}")
    raise


# 定义请求模型
class ImageRequest(BaseModel):
    image: str


@app.get("/")
async def root():
    """
    测试接口是否正常运行
    """
    return {"message": "OCR service is running"}


@app.get("/ocr")
async def read_image_get():
    """
    测试OCR接口是否可访问
    """
    return {"status": "OCR endpoint is working"}


# 修正Base64字符串的填充
def fix_base64_padding(base64_string):
    padding = len(base64_string) % 4
    if padding:
        base64_string += "=" * (4 - padding)  # 添加填充字符
    return base64_string


@app.post("/ocr")
async def read_image(request: ImageRequest):
    """
    接收base64编码的图片字符串进行OCR识别
    """
    try:
        # 处理可能包含的base64前缀
        image = request.image
        print("Received image:", image)  # 打印接收到的图片数据

        # 解码URL编码
        image = unquote(image)

        # 修正Base64字符串的填充
        image = fix_base64_padding(image)

        # 解码base64图片
        try:
            image_bytes = base64.b64decode(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 decoding failed: {str(e)}")

        print("OCR classification started")
        result = ocr.classification(image_bytes)
        print("OCR result:", result)  # 打印OCR结果
        if not result:
            raise HTTPException(status_code=400, detail="OCR recognition failed, no result returned.")

        return {"result": result}

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # 打印错误日志
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 80))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="auto",
        timeout_keep_alive=65,
        access_log=True,
        log_level="info"
    )

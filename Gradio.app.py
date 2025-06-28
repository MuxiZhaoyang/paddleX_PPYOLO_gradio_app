import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import traceback

import subprocess
import sys
import importlib.util

def install_paddlex():
    try:
        # 执行pip install命令安装PaddleX
        print("正在安装PaddleX...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlex"])
        print("PaddleX安装完成。")
        
        # 检查PaddlePaddle是否正确安装
        print("正在检查PaddlePaddle安装...")
        spec = importlib.util.find_spec("paddle")
        if spec is None:
            raise ModuleNotFoundError("PaddlePaddle未安装或安装失败")
        
        # 检查PaddleX是否正确安装
        print("正在检查PaddleX安装...")
        spec = importlib.util.find_spec("paddlex")
        if spec is None:
            raise ModuleNotFoundError("PaddleX未安装或安装失败")
        
        # 打印版本信息（可选）
        import paddle
        import paddlex
        print(f"PaddlePaddle版本: {paddle.__version__}")
        print(f"PaddleX版本: {paddlex.__version__}")
        
        print("环境检查成功，PaddleX已正确安装。")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"安装过程中发生错误: {e}")
        return False
    except ModuleNotFoundError as e:
        print(f"模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

# 执行安装和检查
if __name__ == "__main__":
    success = install_paddlex()
    if success:
        print("安装和检查流程已成功完成。")
    else:
        print("安装或检查过程中出现问题，请查看上面的错误信息。")

from paddlex import create_pipeline
print("正在启动 Gradio 应用...")
# print(f"Gradio 版本: {gr.__version__}")

# 加载模型
print("正在加载PP-YOLOE_plus-S模型...")
# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)

# 输出当前脚本的位置
print(f"当前脚本的位置是: {current_file_path}")
try:
    # 直接在代码中定义配置字典
    # 注意：请确保 model_dir 的路径正确
    config = {
        "pipeline_name": "object_detection",
        "SubModules": {
            "ObjectDetection": {
                "module_name": "object_detection",
                "model_name": "PP-YOLOE_plus-S",
                #"model_dir": r"C:\Users\23295\Desktop\inference",
                #"model_dir": r"C:\ProgramCodeFiles\DL\CURSOR\gradio_paddle\inference",
                #"model_dir": r"gradio_paddle\inference",
                #"model_dir": r"\gradio_paddle\inference",
                "model_dir": r"\home\aistudio\inference",
                "batch_size": 1,
                "threshold": 0.5
            }
        }
    }
    print("使用以下配置加载模型：")
    print(json.dumps(config, indent=2))

    # 通过配置字典加载模型产线
    pipeline = create_pipeline(config=config)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    traceback.print_exc()
    pipeline = None

def detect_objects(img_input):
    """
    目标检测函数
    Args:
        img_input: 输入图片 (PIL Image 或 numpy array)
    Returns:
        检测结果图片 (numpy array)
    """
    if pipeline is None:
        print("模型未加载，返回原图")
        return img_input
    
    try:
        print("开始进行目标检测...")
        
        # 确保输入是numpy数组格式
        if hasattr(img_input, 'shape'):
            input_img = img_input
        else:
            input_img = np.array(img_input)
        
        # Gradio 的 Image 组件输出的是 RGB 格式，而 PaddleX 和 OpenCV 通常使用 BGR 格式。
        # 在将图像送入模型之前，需要进行颜色通道转换。
        img_bgr = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)

        # 进行推理，结果是一个生成器
        prediction_generator = pipeline.predict(img_bgr)
        
        # 将生成器转换为列表
        results = list(prediction_generator)

        # 获取检测结果
        if results and len(results) > 0:
            result = results[0]
            print(f"原始检测结果: {result}")
            
            # 如果有检测到目标，绘制边界框和标签
            if 'boxes' in result and len(result['boxes']) > 0:
                # BGR to RGB, then to PIL Image for drawing
                pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 定义一个颜色列表 (RGB格式)
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (192, 192, 192),
                    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
                    (128, 0, 128), (0, 128, 128)
                ]
                
                # 自动搜索并加载中文字体
                font_path_list = [
                    "C:/Windows/Fonts/msyh.ttf",   # 微软雅黑
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/deng.ttf",    # 等线
                ]
                font = None
                font_size = 15
                
                for font_path in font_path_list:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        print(f"成功加载字体: {font_path}")
                        break
                    except IOError:
                        continue # 字体不存在或无法加载，尝试下一个
                
                if font is None:
                    print("警告: 未在系统中找到可用的中文字体。将使用默认字体，中文可能无法正常显示。")
                    font = ImageFont.load_default()

                for box in result['boxes']:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box['coordinate'])
                    
                    # 获取类别、置信度和类别ID
                    label = box['label']
                    score = box['score']
                    cls_id = box['cls_id']
                    
                    # 根据类别ID选择颜色
                    color = colors[cls_id % len(colors)]
                    
                    # 确定文本颜色（深色背景用白色字，浅色背景用黑色字）
                    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)
                    
                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 准备标签文本
                    text = f"{label}: {score:.2f}"
                    
                    # Pillow 9.2.0+ anker
                    if hasattr(draw, 'textbbox'):
                        # 绘制标签背景
                        text_bbox = draw.textbbox((x1, y1 - font_size - 4), text, font=font)
                        # 确保背景框不会超出图片顶部
                        text_bbox = (text_bbox[0], max(0, text_bbox[1]), text_bbox[2], max(font_size, text_bbox[3]))
                        draw.rectangle(text_bbox, fill=color)
                        # 绘制标签文本
                        draw.text((x1, text_bbox[1]), text, font=font, fill=text_color)
                    else:
                         # 对于旧版Pillow，没有背景，直接用对比色绘制文本
                         draw.text((x1, y1 - font_size - 4), text, font=font, fill=text_color)

                print(f"检测到 {len(result['boxes'])} 个目标")
                
                # 将PIL图像转换回Gradio所需的numpy数组 (RGB)
                return np.array(pil_img)
            else:
                print("未检测到任何目标")
                return input_img
        else:
            print("推理结果为空，返回原图")
            return input_img
            
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        traceback.print_exc()
        return img_input

# --- Gradio UI 美化 ---
css = """
body { background-color: #f0f2f6; }
.gradio-container { max-width: 1100px !important; margin: auto !important; padding-top: 2rem !important; }
#title { text-align: center; font-size: 2.2rem; font-weight: bold; color: #2c3e50; }
#subtitle { text-align: center; color: #57606f; margin-top: -1rem; margin-bottom: 2rem; }
.gr-button-primary { background: linear-gradient(to right, #4facfe, #00f2fe) !important; border: none !important; font-size: 1.1rem !important; padding: 12px !important; }
footer { display: none !important }
.image-display > div {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    height: 100% !important;
    width: 100% !important;
}
.image-display > div > img {
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
}
"""

# 创建Gradio界面
print("创建界面...")
with gr.Blocks(css=css, title="PP-YOLOE+ 智能目标检测") as demo:
    gr.Markdown("# PP-YOLOE+ 智能目标检测", elem_id="title")
    gr.Markdown("✨ 由 PaddleX 强力驱动，高效、精准、美观 ✨", elem_id="subtitle")
    
    with gr.Row(variant="panel"):
        input_image = gr.Image(label="🖼️ 上传图片", type="numpy", elem_classes="image-display")
        output_image = gr.Image(label="💡 检测结果", type="numpy", elem_classes="image-display")
    
    with gr.Row():
        detect_btn = gr.Button("🚀 开始检测", variant="primary")
    
    # 设置点击事件
    detect_btn.click(
        fn=detect_objects,
        inputs=input_image,
        outputs=output_image
    )
    
    # 也可以设置自动检测（图片上传后自动处理）
    input_image.change(
        fn=detect_objects,
        inputs=input_image,
        outputs=output_image
    )

print("启动应用...")
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
print("应用已启动")
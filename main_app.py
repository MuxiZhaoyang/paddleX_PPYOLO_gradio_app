# import gradio as gr
# print("正在启动 Gradio 应用...")
# print(f"Gradio 版本: {gr.__version__}")

# def demo(img_input):
#     print("收到图片输入")
#     img_output=img_input
#     return img_output

# print("创建界面...")
# demo = gr.Interface(fn=demo, 
#                     inputs="image", 
#                     outputs=gr.Image(type="numpy")
#                     )     

# print("启动应用...")
# demo.launch()
# print("应用已启动")


import gradio as gr
import cv2
import numpy as np
from paddlex import create_pipeline, create_predictor
import os
import shutil

print("正在启动 Gradio 应用...")
# print(f"Gradio 版本: {gr.__version__}")

# 模型路径
MODEL_PATH = "gradio_paddle/inference"

# 加载模型
print("正在加载PP-YOLOE_plus-S模型...")
try:
    # 适配 PaddleX 3.0 的模型文件命名
    # 检查旧版文件是否存在并重命名
    print("正在检查并适配模型文件...")
    old_yml_path = os.path.join(MODEL_PATH, "inference.yml")
    new_yml_path = os.path.join(MODEL_PATH, "infer_cfg.yml")
    if os.path.exists(old_yml_path) and not os.path.exists(new_yml_path):
        shutil.copyfile(old_yml_path, new_yml_path)
        print(f"已将 {old_yml_path} 复制为 {new_yml_path}")

    # 检查是否存在 .pdiparams 文件但缺少 .pdmodel 文件
    pdiparams_path = os.path.join(MODEL_PATH, "inference.pdiparams")
    pdmodel_path = os.path.join(MODEL_PATH, "model.pdmodel")
    if os.path.exists(pdiparams_path) and not os.path.exists(pdmodel_path):
        # 如果缺少 .pdmodel，可能需要重新导出模型。
        # 这里我们假设 inference.json 可能是模型结构文件，尝试重命名它
        # 注意：这只是一个基于常见情况的猜测
        json_path = os.path.join(MODEL_PATH, "inference.json")
        if os.path.exists(json_path):
            shutil.copyfile(json_path, pdmodel_path)
            print(f"警告：缺少 'model.pdmodel'。已尝试将 {json_path} 复制为 {pdmodel_path}")

    # 重命名权重文件
    new_pdiparams_path = os.path.join(MODEL_PATH, "model.pdiparams")
    if os.path.exists(pdiparams_path) and not os.path.exists(new_pdiparams_path):
         shutil.copyfile(pdiparams_path, new_pdiparams_path)
         print(f"已将 {pdiparams_path} 复制为 {new_pdiparams_path}")


    # 在新版本中，需要先创建 predictor
    predictor = create_predictor(MODEL_PATH)
    # 然后使用 predictor 创建管道
    pipeline = create_pipeline("object_detection", predictor=predictor)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
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
        
        # 进行推理
        results = pipeline.predict(input_img)
        
        # 获取检测结果
        if results and len(results) > 0:
            result = results[0]
            
            # 在图片上绘制检测框
            output_img = result.img.copy()
            
            # 如果有检测到目标，绘制边界框和标签
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box['coordinate'])
                    
                    # 获取类别和置信度
                    label = box['label']
                    score = box['score']
                    
                    # 绘制边界框
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制标签和置信度
                    text = f"{label}: {score:.2f}"
                    cv2.putText(output_img, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"检测到 {len(result.boxes)} 个目标")
            else:
                print("未检测到任何目标")
            
            return output_img
        else:
            print("推理结果为空，返回原图")
            return input_img
            
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        return img_input

# 创建Gradio界面
print("创建界面...")
with gr.Blocks(title="PP-YOLOE_plus-S 目标检测") as demo:
    gr.Markdown("# PP-YOLOE_plus-S 目标检测应用")
    gr.Markdown("上传一张图片，模型将检测图片中的目标并标注出来")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 输入图片")
            input_image = gr.Image(label="上传图片", type="numpy")
            detect_btn = gr.Button("开始检测", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 检测结果")
            output_image = gr.Image(label="检测结果", type="numpy")
    
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
# 第一步：克隆应用空间

```linux
git lfs install
git clone http://git.aistudio.baidu.com/11670913/gradio_paddle.git
```

[获取access token](https://aistudio.baidu.com/account/accessToken)

# 第二步：创建Gradio.app.py文件

```python
import gradio as gr

def quickstart(name):
    return "欢迎使用飞桨AI Studio应用创建工具Gradio, " + name + "!!!"
demo = gr.Interface(fn=quickstart, inputs="text", outputs="text")

demo.launch()

```

# 第三步：上传&提交文件

```
git add Gradio.app.py
git commit -m "Add application file"
git push
```

# 从项目中部署

除了在本地完成代码开发，上传至应用空间之外，我们也允许大家在[项目](https://aistudio.baidu.com/my/project)中编辑Gradio前端代码，编辑完成后点击部署即可自动创建应用空间并完成发布。



Gradio 版本支持：4.26.0、4.0.0

完整文档请见：https://ai.baidu.com/ai-doc/AISTUDIO/Plu48z144 
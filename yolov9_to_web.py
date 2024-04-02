import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO

# 设置 Streamlit 页面标题
st.title('YOLOv9 物体检测')

# 设置文件上传器，让用户可以上传图片
uploaded_file = st.file_uploader("请选择一个图片文件", type=['jpg', 'png', 'jpeg'])

# 检查是否有文件被上传
if uploaded_file is not None:
    # 打开上传的图片文件
    image = Image.open(uploaded_file)
    # 显示上传的图片
    st.image(image, caption='上传的图片', use_column_width=True)
    
    # 加载模型，注意这一行可能需要根据实际仓库的代码结构进行调整
    # 并且确保 streamlit 能够在服务器上执行外部网络请求
    model = torch.hub.load('wongkinyiu/yolov9', 'yolov9', source='github', trust_repo=True)
    
    # 图片预处理，可能需要根据 YOLOv9 模型的具体要求调整
    transform = transforms.Compose([
        transforms.Resize(640),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_img = transform(image).unsqueeze(0)
    
    # 设置为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 进行预测
        results = model(input_img)
    
    # 绘制检测的边界框和标签，这需要根据模型返回的结果格式调整
    # 这里只是提供一个示例框架，你可能需要修改它以适应 YOLOv9 的输出
    results.render()  # 更新 input_img 以显示边界框和标签
    rendered_img = Image.fromarray(input_img.mul(255).permute(1, 2, 0).byte().numpy())
    img_bytes = BytesIO()
    rendered_img.save(img_bytes, format='JPEG')
    # 显示带有检测结果的图片
    st.image(img_bytes, caption='检测结果', use_column_width=True)

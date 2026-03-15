"""
广义指代图像分割 (G-RIS) 演示系统
基于CRIS-P模型，支持文本引导的图像分割
"""

import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

from model import build_segmenter
from utils.dataset import tokenize
from utils.config import load_cfg_from_cfg_file


class GRISDemo:
    """G-RIS演示类"""

    def __init__(self, model_path, config_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载配置
        cfg = load_cfg_from_cfg_file(config_path)
        cfg.clip_pretrain = model_path.replace("\\", "/")

        # 构建模型
        self.model, _ = build_segmenter(cfg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载训练好的权重
        if os.path.exists("exp/refcoco/CRIS_R50/best_model.pth"):
            checkpoint = torch.load("exp/refcoco/CRIS_R50/best_model.pth",
                                    map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("加载最佳模型权重成功")
        else:
            print("使用预训练CLIP权重")

        self.word_len = cfg.word_len

    def segment(self, image, text):
        """
        对图像进行指代分割

        Args:
            image: PIL Image
            text: 文本描述

        Returns:
            result_image: 分割结果图像
        """
        if image is None or text is None:
            return None

        # 转换为numpy数组
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 预处理
        h, w = img.shape[:2]
        input_size = 416
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (new_w, new_h))
        img_padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        img_padded[:new_h, :new_w] = img_resized

        # 归一化
        mean = np.array([0.48145466, 0.4578275, 0.40821073]) * 255
        std = np.array([0.26862954, 0.26130258, 0.27577711]) * 255
        img_normalized = (img_padded - mean) / std
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()

        # 文本编码
        word = tokenize(text, self.word_len, True)

        # 推理
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            word = word.to(self.device)
            pred = self.model(img_tensor, word)
            pred = torch.sigmoid(pred)

        # 后处理
        pred = pred.squeeze().cpu().numpy()
        pred = cv2.resize(pred, (w, h))
        mask = (pred > 0.35).astype(np.uint8) * 255

        # 绘制结果
        result = image.copy()
        result = np.array(result)

        # 创建彩色mask
        mask_colored = np.zeros_like(result)
        mask_colored[:, :, 1] = mask  # 绿色

        # 叠加
        result = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)

        # 添加文字标注
        cv2.putText(result, text[:30], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result


def create_demo():
    """创建Gradio演示界面"""

    # 模型路径
    model_path = "pretrain/RN50.pt"
    config_path = "config/refcoco/cris_r50.yaml"

    # 如果没有训练好的模型，使用预训练权重
    demo = GRISDemo(model_path, config_path)

    # 创建界面
    with gr.Blocks(title="G-RIS 广义指代图像分割") as demo:
        gr.Markdown("# 广义指代图像分割 (G-RIS)")
        gr.Markdown("基于跨模态对齐的指代图像分割算法 - 上传图片并输入文本描述进行分割")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="输入图像")
                input_text = gr.Textbox(label="文本描述",
                                        placeholder="例如: the cat on the left",
                                        value="the white square")
                submit_btn = gr.Button("分割", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="分割结果")

        submit_btn.click(
            fn=demo.segment,
            inputs=[input_image, input_text],
            outputs=output_image
        )

        # 示例
        gr.Examples(
            examples=[
                ["examples/cat.jpg", "the cat"],
                ["examples/dog.jpg", "the white dog"],
            ],
            inputs=[input_image, input_text]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)

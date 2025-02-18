import onnxruntime as ort
import cv2
import numpy as np
import time

class ONNXPredictor:
    def __init__(self, onnx_path):
        # 初始化运行时
        self.session = ort.InferenceSession(
            onnx_path,
            providers=[
                'AzureExecutionProvider',  # GPU优先
                'CPUExecutionProvider'     # 备用CPU
            ]
        )
        self.input_name = self.session.get_inputs()[0].name
        # 预处理参数（与训练一致）
        self.mean = 0.5
        self.std = 0.5

    def remove_black_lines(self,image_path):
        # # 读取图像
        # img = cv2.imread(image_path)
        img = image_path
        # # 创建黑色像素掩膜（RGB都小于10）
        mask = np.all(img < [50, 50, 50], axis=2).astype(np.uint8) * 255
        # # 使用图像修复去除干扰线
        result = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

        return result
    
    def preprocess(self, image_path):
        """与之前相同的预处理流程"""
        img = cv2.imread(image_path)
        cleaned = self.remove_black_lines(img)
        # gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(cleaned, (90, 32))
        resized = resized.transpose((2, 0, 1))  # 转换为 (channels, height, width)
        normalized = (resized / 255.0 - self.mean) / self.std
        return np.expand_dims(normalized, axis=0 ).astype(np.float32)

    def predict(self, image_path):
        input_data = self.preprocess(image_path)
        outputs = self.session.run(None, {self.input_name: input_data})[0]
        preds = np.argmax(outputs, axis=2)
        return decode_prediction(preds[0])

def decode_prediction(pred_ids):
    characters = [chr(ord('a') + i) for i in range(26)] + [str(i) for i in range(10)] 
    return ''.join([characters[idx] for idx in pred_ids])


# 开始计时
start_time = time.perf_counter()
# 使用示例
predictor = ONNXPredictor("0.9997.onnx")
result = predictor.predict("captcha2.jpg")
print("识别结果:", result)
# 结束计时
end_time = time.perf_counter()
# 计算执行时间
elapsed_time = end_time - start_time
print(f"代码执行时间：{elapsed_time:.6f} 秒")
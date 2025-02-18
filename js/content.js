  // ------------------------- 核心代码 (content.js) -------------------------
  // 依赖库：需要包含以下脚本（在manifest中注入）
  // <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
  // <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

let isProcessing = false;
let session = null;
let autoSubmitEnabled = false; // 自动提交状态
let submitButtonObserver = null; // 观察提交按钮状态

// 初始化ONNX会话
async function initModel() {
try {
    const modelPath = chrome.runtime.getURL('model/captcha.onnx');
    session = await ort.InferenceSession.create(modelPath);
    console.log('ONNX模型加载成功');
} catch (error) {
    console.error('模型加载失败:', error);
}
}

// 加载自动提交配置
chrome.storage.local.get(['autoSubmit'], (result) => {
    autoSubmitEnabled = !!result.autoSubmit;
    console.log('自动提交状态:', autoSubmitEnabled ? '开启' : '关闭');
});

  // 监听配置变化
chrome.storage.onChanged.addListener((changes) => {
    if (changes.autoSubmit) {
    autoSubmitEnabled = changes.autoSubmit.newValue;
    console.log('自动提交状态更新:', autoSubmitEnabled ? '开启' : '关闭');
    }
});



// 修改后的预处理函数（保持RGB三通道）
function preprocessImage(imageData) {
    const width = 90, height = 32;
    const inputData = new Float32Array(1 * 3 * height * width); // 改为3通道
    
    // 去除黑色干扰线并保持RGB
    for (let i = 0; i < imageData.length; i += 4) {
    let r = imageData[i];
    let g = imageData[i + 1];
    let b = imageData[i + 2];
    
      // 去除黑色干扰线（RGB均<50）
    if (r < 50 && g < 50 && b < 50) {
        r = g = b = 255;
    }
    
      // 归一化处理（与训练时保持一致）
    const normalized = [
        (r / 255.0 - 0.5) / 0.5, // R通道
        (g / 255.0 - 0.5) / 0.5, // G通道
        (b / 255.0 - 0.5) / 0.5  // B通道
    ];
    
      // 转换为CHW格式 (1x3x32x90)
    const x = Math.floor((i/4) % width);
    const y = Math.floor((i/4) / width);
    
    inputData[y * width + x] = normalized[0];               // R通道
    inputData[width*height + y*width + x] = normalized[1];  // G通道
    inputData[2*width*height + y*width + x] = normalized[2];  // B通道
    }
    
    return inputData;
}

// 执行识别
async function recognizeCaptcha(imgElement) {
if (!session || isProcessing) return;

isProcessing = true;

try {
    // 创建Canvas处理图像
    const canvas = document.createElement('canvas');
    canvas.width = imgElement.naturalWidth;
    canvas.height = imgElement.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imgElement, 0, 0);
    
    // 预处理
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const inputData = preprocessImage(imageData);
    
    // 创建输入张量
    const tensor = new ort.Tensor('float32', inputData, [1, 3, 32, 90]);
    
    // 执行推理
    const results = await session.run({ input: tensor });
    const output = results.output.data;
    
    // 解码结果
    const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
    let captcha = '';
    for (let i = 0; i < 4; i++) {
      const start = i * 36;
    const maxIndex = output.indexOf(Math.max(...output.slice(start, start+36)));
    captcha += chars[maxIndex % 36];
    }
    
    // 填充结果
    const inputField = document.getElementById('captcha');
    if (inputField) inputField.value = captcha;
    console.log('验证码:', captcha);
    // 启用提交按钮
    const submitBtn = document.getElementsByName('submit')[0];
    if (submitBtn) submitBtn.disabled = false;


} catch (error) {
    console.error('识别失败:', error);
} finally {
    isProcessing = false;
}
}

// 监听验证码图片变化
function setupImageObserver() {
const targetNode = document.getElementById('captchaImg');
if (!targetNode) return;

  // 立即执行首次识别
recognizeCaptcha(targetNode);

  // 点击刷新时重新识别
targetNode.addEventListener('click', () => {
  const submitBtn = document.getElementsByName('submit')[0];
    setTimeout(() => {
    recognizeCaptcha(targetNode);
    if (autoSubmitEnabled){
    submitBtn.click()
    };
}, 500);
});
}

// 初始化
(async () => {
await initModel();
setupImageObserver();

  // 轮询检查元素（应对动态加载）
const checkInterval = setInterval(() => {
    if (document.getElementById('captchaImg')) {
    clearInterval(checkInterval);
    setupImageObserver();
    }
}, 500);
})();


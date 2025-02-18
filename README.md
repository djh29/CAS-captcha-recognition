# CAS captcha

---

1、该扩展实现以下功能：
  
- 加载 SYSU CAS 登录网站时，自动识别验证码并填写到验证码输入框，点击验证码可重新识别
- 扩展有个自动登录开关，默认关闭，如果打开，验证码输入后，点击键盘任意键即可自动登录（不限于回车键）。

---

2、模型使用`CNN`训练，准确率接近`99.97%`。另外，模型大小小于1M,识别速度小于`50ms`。

---

3、使用onnxruntime-web加载模型。在浏览器`扩展管理`，打开`开发人员模式`，点击`加载解压缩的扩展`，选择扩展的`整个文件夹`，即把扩展安装在浏览器上。

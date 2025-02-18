// 后台服务（Manifest V3）
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({ autoSubmit: false });
});

// 接收来自内容脚本的消息
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'getAutoSubmitStatus') {
    chrome.storage.local.get(['autoSubmit'], result => {
      sendResponse({ autoSubmit: result.autoSubmit });
    });
    return true; // 保持异步响应
  }
});
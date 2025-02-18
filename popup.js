document.addEventListener('DOMContentLoaded', () => {
  const autoSubmit = document.getElementById('autoSubmit');

  // 加载保存的设置
  chrome.storage.local.get(['autoSubmit'], result => {
    autoSubmit.checked = !!result.autoSubmit;
  });

  // 保存设置变更
  autoSubmit.addEventListener('change', () => {
    chrome.storage.local.set({ autoSubmit: autoSubmit.checked });
  });
});
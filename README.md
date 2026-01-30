# 🌿 中医智能小助手（Streamlit Web 原型）

一个用 Streamlit 搭建的“中医智能小助手”原型：先让用户用日常语言自由描述不适，再通过追问补全信息，最后输出 **状态分析 + 养生建议**。

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Configure DeepSeek API key

   Create `.streamlit/secrets.toml` (this file is gitignored)：

   ```
   DEEPSEEK_API_KEY="你的DeepSeek_API_Key"
   ```

3. Run the app

   ```
   $ python -m streamlit run streamlit_app.py
   ```

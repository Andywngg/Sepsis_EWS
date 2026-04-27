# if you use the venv
.\.venv\Scripts\activate

# install demo deps (once)
pip install -r requirements_demo.txt

# run
streamlit run demo_app/app.py

If streamlit isn’t recognized, use:
.\.venv\Scripts\python -m streamlit run demo_app/app.py

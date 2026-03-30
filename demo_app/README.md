# Sepsis EWS Demo (Local)

This is a local Streamlit demo for presenting the model to judges.

## Install
From `sepsis_ews/`:

```powershell
pip install -r requirements_demo.txt
```

## Run
```powershell
streamlit run demo_app/app.py
```

### Share on Local Network (QR Code)
Run with a network-visible address so judges can open the demo on their phones.

```powershell
streamlit run demo_app/app.py --server.address 0.0.0.0 --server.port 8501
```

Then enter your laptop's local IP (for example, `http://192.168.x.x:8501`) into the app's Share section to generate a QR code.

## Tips
- Use the filter box to find a specific patient ID.
- Adjust the threshold and alert_k to show tradeoffs in real time.
- Show the lead time between the first alert and sepsis onset.

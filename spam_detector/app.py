import os
import re
import joblib
import streamlit as st

# Check if model files exist
if not (os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl')):
    import subprocess
    subprocess.run(["python", "train_model.py"])

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title="Advanced Spam Detector", layout="wide")

st.title("📧 Spam Email Detector")
st.write("Analyze email content or full email headers to check for spam, sender info, and location details.")

# Header tabs
tab1, tab2 = st.tabs(["🔍 Quick Email Text Check", "📑 Deep Header & Metadata Analysis"])

with tab1:
    user_input = st.text_area("Paste email text here:", height=150)
    if st.button("Check Email Text"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            vec = vectorizer.transform([user_input])
            pred = model.predict(vec)[0]
            if pred == 1 or str(pred).lower() == 'spam':
                st.error("🚨 SPAM DETECTED!")
            else:
                st.success("✅ SAFE / NOT SPAM (HAM)")

with tab2:
    st.subheader("Extract Sender, Date & IP Location from Headers")
    raw_header = st.text_area(
        "Paste Full Email Header here:",
        height=200,
        placeholder="Paste text copied from Gmail 'Show original' here..."
    )
    
    if st.button("Extract Header Metadata"):
        if not raw_header.strip():
            st.warning("Please paste header text.")
        else:
            from_match = re.search(r"From:\s*(.*)", raw_header, re.IGNORECASE)
            sender = from_match.group(1).strip() if from_match else "Unknown Sender"

            date_match = re.search(r"Date:\s*(.*)", raw_header, re.IGNORECASE)
            date_sent = date_match.group(1).strip() if date_match else "Unknown Date"

            ip_matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw_header)
            public_ips = [ip for ip in ip_matches if not (ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("127."))]
            origin_ip = public_ips[0] if public_ips else ("Local IP (" + ip_matches[0] + ")" if ip_matches else "Not Found")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="👤 From (Sender)", value=sender)
            with col2:
                st.metric(label="📅 Date Received", value=date_sent)
            with col3:
                st.metric(label="🌐 Originating IP", value=origin_ip)

            st.divider()

            vec = vectorizer.transform([raw_header])
            pred = model.predict(vec)[0]
            if pred == 1 or str(pred).lower() == 'spam':
                st.error("🚨 Content Assessment: SPAM")
            else:
                st.success("✅ Content Assessment: SAFE")
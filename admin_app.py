"""SafeChat-AI Admin Dashboard (Streamlit).

Run with:  streamlit run admin_app.py
"""
import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="SafeChat-AI Admin", page_icon="\U0001F6E1\uFE0F", layout="wide")
st.title("SafeChat-AI Admin Dashboard")

VIOLATIONS_FILE = Path("models/violations.json")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Violations Overview")
    if VIOLATIONS_FILE.exists():
        try:
            data = json.loads(VIOLATIONS_FILE.read_text())
            if data:
                rows = []
                for uid, rec in data.items():
                    rows.append({
                        "User": uid,
                        "Violations": len(rec.get("violations", [])),
                        "Blocked Until": (
                            pd.to_datetime(rec.get("blocked_until", 0), unit="s")
                            if rec.get("blocked_until", 0) else "N/A"
                        ),
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                if st.button("Reset All Users", type="primary"):
                    VIOLATIONS_FILE.write_text("{}")
                    st.success("All violations reset.")
                    st.rerun()
            else:
                st.info("No violations recorded.")
        except Exception as e:
            st.error(f"Error reading violations: {e}")
    else:
        st.info("No violations file found.")

with col2:
    st.subheader("Model Info")
    st.markdown("**Default Model:** `unitary/toxic-bert`")
    st.markdown("**Fine-tuned Model:** `models/bert_cyberbully`")
    st.markdown("---")
    st.subheader("Quick Actions")
    if st.button("View API Logs (last 100 lines)"):
        st.code("Run: docker logs safechat-api --tail 100", language="bash")
    if st.button("Run Tests"):
        st.code("pytest tests/ -v", language="bash")

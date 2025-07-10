# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
#
# # Load model and preprocessing objects
# model = load_model("ticket_model.h5")
#
# with open("tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)
#
# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)
#
# # App title and description
# st.set_page_config(page_title="🎫 IT Ticket Classifier", layout="centered")
#
# st.markdown("""
#     <h2 style='text-align: center; color: #3366cc;'>NLP-Based Ticket Classification</h2>
#     <p style='text-align: center; color: #555;'>
#         Classify IT service tickets automatically into Oracle-related modules using an AI-powered model.
#     </p>
# """, unsafe_allow_html=True)
#
# st.markdown("---")
#
# # Input section
# with st.container():
#     st.markdown("#### 📝 Enter Ticket Description")
#     ticket_text = st.text_area("Describe the issue you're facing:", height=150, placeholder="e.g. Unable to post invoices in Oracle Payables...")
#
#     submit = st.button("🔍 Classify Ticket")
#
# # Prediction
# if submit and ticket_text.strip():
#     seq = tokenizer.texts_to_sequences([ticket_text])
#     padded = pad_sequences(seq, maxlen=20)
#     pred = model.predict(padded)
#     predicted_label = label_encoder.inverse_transform([np.argmax(pred)])
#
#     st.markdown("---")
#     st.markdown(f"""
#     <div style='background-color: #e6f2ff; padding: 20px; border-radius: 10px;'>
#         <h4>✅ Predicted Module:</h4>
#         <h2 style='color: #004080;'>{predicted_label[0]}</h2>
#     </div>
#     """, unsafe_allow_html=True)
#
# elif submit:
#     st.warning("⚠️ Please enter a ticket description to classify.")
#
# # Footer
# st.markdown("---")
# st.markdown("""
# <p style='text-align: center; font-size: 13px; color: gray;'>
#     Built with ❤️ using TensorFlow, Streamlit, and NLP.
# </p>
# """, unsafe_allow_html=True)
#
# import streamlit as st
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
#
# # Load assets
# model = load_model("ticket_model.h5")
# tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
# label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
#
# st.title("🎫 IT Ticket Classifier")
# st.subheader("Enter Ticket Description")
# text = st.text_area("Describe the issue you're facing:")
#
# if st.button("🔍 Classify Ticket"):
#     if text.strip() == "":
#         st.warning("Please enter a ticket description.")
#     else:
#         seq = tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=20, padding='post')
#         prediction = model.predict(padded)
#         predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
#         st.success(f"✅ Predicted Module:\n\n### {predicted_class[0]}")

import streamlit as st
import pickle
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("🎫 IT Ticket Classifier")
st.subheader("Enter Ticket Description")

# Safely load model and preprocessing tools
try:
    model = load_model("ticket_model.keras", compile=False)  # safer than .h5
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Failed to load model or preprocessing files.\n\nError: {e}")
    sys.exit(1)

text = st.text_area("Describe the issue you're facing:")

if st.button("🔍 Classify Ticket"):
    if text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=20, padding='post')
        prediction = model.predict(padded)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        st.success(f"✅ Predicted Module:\n\n### {predicted_class[0]}")

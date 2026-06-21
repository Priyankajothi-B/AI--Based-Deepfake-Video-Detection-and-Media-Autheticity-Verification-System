import streamlit as st
import sys
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Deepfake Detection & Media Authenticity Verification",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle {
    text-align:center;
    color:white;
    font-size:18px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background:#1e293b;
    color:white;
    margin-bottom:15px;
}

.metric-card{
    background:#111827;
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# BASE DIRECTORY
# =====================================================

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# =====================================================
# IMPORT MODULES
# =====================================================

try:

    sys.path.append(
        os.path.join(
            base_dir,
            "Deepfake_Detection_Module",
            "src"
        )
    )

    from deepfake_detector import (
        get_deepfake_result_from_upload
    )

    deepfake_loaded = True

except Exception as e:

    st.error(f"Deepfake Module Error : {e}")
    deepfake_loaded = False

try:

    sys.path.append(
        os.path.join(
            base_dir,
            "metadata_analysis",
            "src"
        )
    )

    from metadata_analyzer import (
        analyze_uploaded_video
    )

    metadata_loaded = True

except Exception as e:

    st.error(f"Metadata Module Error : {e}")
    metadata_loaded = False

try:

    sys.path.append(
        os.path.join(
            base_dir,
            "gradcam_explainability",
            "src"
        )
    )

    from gradcam_generator import (
        generate_gradcam_from_upload
    )

    gradcam_loaded = True

except Exception as e:

    st.error(f"GradCAM Module Error : {e}")
    gradcam_loaded = False

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='title'>🛡️ AI Deepfake Detection & Media Authenticity Verification System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Final Year Project | Deepfake Detection + Metadata Analysis + Explainable AI</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 System Status")

st.sidebar.write(
    "Deepfake Module",
    "✅" if deepfake_loaded else "❌"
)

st.sidebar.write(
    "Metadata Module",
    "✅" if metadata_loaded else "❌"
)

st.sidebar.write(
    "GradCAM Module",
    "✅" if gradcam_loaded else "❌"
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Upload a video and the system will:

    • Detect Deepfakes

    • Verify Metadata

    • Generate Explainability Heatmaps

    • Produce Final Verdict
    """
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "📂 Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

# =====================================================
# MAIN ANALYSIS
# =====================================================

if uploaded_file:

    st.success("Video Uploaded Successfully")

    st.video(uploaded_file)

    if st.button("🚀 Analyze Video"):

        st.markdown("---")

        # =============================================
        # DEEPFAKE ANALYSIS
        # =============================================

        if deepfake_loaded:

            with st.spinner("Running Deepfake Detection..."):

                deepfake_result = get_deepfake_result_from_upload(
                    uploaded_file
                )

            st.subheader("🎭 Deepfake Detection")

            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Deepfake Score",
                f"{deepfake_result['deepfake_score']}%"
            )

            col2.metric(
                "Verdict",
                deepfake_result["verdict"]
            )

            col3.metric(
                "Risk Level",
                deepfake_result["risk_level"]
            )

        # =============================================
        # RESET FILE POINTER
        # =============================================

        uploaded_file.seek(0)

        # =============================================
        # METADATA ANALYSIS
        # =============================================

        if metadata_loaded:

            with st.spinner("Analyzing Metadata..."):

                metadata_result = analyze_uploaded_video(
                    uploaded_file
                )

            st.subheader("📑 Metadata Analysis")

            st.json(metadata_result)

        # =============================================
        # RESET FILE POINTER
        # =============================================

        uploaded_file.seek(0)

        # =============================================
        # GRADCAM
        # =============================================

        if gradcam_loaded:

            with st.spinner("Generating GradCAM Heatmap..."):

                gradcam_result = generate_gradcam_from_upload(
                    uploaded_file,
                    os.path.join(
                        base_dir,
                        "gradcam_explainability",
                        "models",
                        "best_model-v3.pt"
                    )
                )

            st.subheader("🔥 Explainable AI (GradCAM)")

            if gradcam_result["gradcam_frame"] is not None:

                st.image(
                    gradcam_result["gradcam_frame"],
                    caption="GradCAM Heatmap",
                    use_container_width=True
                )

            else:

                st.warning(
                    "Unable to generate heatmap."
                )

        # =============================================
        # FINAL REPORT
        # =============================================

        st.markdown("---")

        st.subheader("📋 Final Decision")

        if deepfake_loaded:

            verdict = deepfake_result["verdict"]

            if verdict == "FAKE":

                st.error(
                    "🚨 HIGH RISK : Deepfake Content Detected"
                )

            elif verdict == "SUSPICIOUS":

                st.warning(
                    "⚠️ SUSPICIOUS CONTENT DETECTED"
                )

            else:

                st.success(
                    "✅ VIDEO APPEARS AUTHENTIC"
                )
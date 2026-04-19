import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import re
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.sidebar.title("🧠 Brain Tumor Detection")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict", "Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates a CNN model trained to classify brain MRI scans "
    "into 4 categories: Glioma, Meningioma, No Tumor, and Pituitary tumors."
)

# Model and data paths
MODEL_PATH = "results/best_model.keras"
RESULTS_PATH = "results"
NOTEBOOK_05_PATH = "notebooks/05_model_improvement.ipynb"
FINAL_HISTORY_PATH = f"{RESULTS_PATH}/final_training_history.npy"
FINAL_METRICS_PATH = f"{RESULTS_PATH}/final_test_metrics.npz"
FINAL_REPORT_PATH = f"{RESULTS_PATH}/final_classification_report.txt"
FINAL_CONFUSION_MATRIX_PATH = f"{RESULTS_PATH}/final_confusion_matrix.npy"

# Cache model loading
@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        return None

@st.cache_data
def load_test_data():
    """Load preprocessed test data"""
    try:
        X_test = np.load(f"{RESULTS_PATH}/X_test.npy")
        y_test = np.load(f"{RESULTS_PATH}/y_test.npy")
        return X_test, y_test
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_training_history():
    """Load training history"""
    try:
        return np.load(f"{RESULTS_PATH}/training_history.npy", allow_pickle=True).item()
    except FileNotFoundError:
        return None

@st.cache_data
def load_notebook05_final_results():
    """Parse final training curves and final test metrics from Notebook 05 outputs."""
    # First, prefer explicit final artifacts exported from Notebook 05.
    if os.path.exists(FINAL_HISTORY_PATH) and os.path.exists(FINAL_METRICS_PATH):
        try:
            history = np.load(FINAL_HISTORY_PATH, allow_pickle=True).item()
            metrics = np.load(FINAL_METRICS_PATH)
            return history, float(metrics["test_loss"]), float(metrics["test_acc"])
        except (OSError, ValueError, KeyError):
            pass

    if not os.path.exists(NOTEBOOK_05_PATH):
        return None, None, None

    try:
        with open(NOTEBOOK_05_PATH, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None, None

    metric_line_pattern = re.compile(
        r"loss:\s*([0-9.]+)\s*-\s*accuracy:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)\s*-\s*val_accuracy:\s*([0-9.]+)"
    )
    test_acc_pattern = re.compile(r"Augmented Model Test Accuracy:\s*([0-9.]+)")
    test_loss_pattern = re.compile(r"Augmented Model Test Loss:\s*([0-9.]+)")

    last_history = None
    last_test_acc = None
    last_test_loss = None

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        found_metric_rows = False

        for output in cell.get("outputs", []):
            text_chunks = output.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]

            for line in text_chunks:
                metric_match = metric_line_pattern.search(line)
                if metric_match:
                    found_metric_rows = True
                    history["loss"].append(float(metric_match.group(1)))
                    history["accuracy"].append(float(metric_match.group(2)))
                    history["val_loss"].append(float(metric_match.group(3)))
                    history["val_accuracy"].append(float(metric_match.group(4)))

                acc_match = test_acc_pattern.search(line)
                if acc_match:
                    last_test_acc = float(acc_match.group(1))

                loss_match = test_loss_pattern.search(line)
                if loss_match:
                    last_test_loss = float(loss_match.group(1))

        if found_metric_rows:
            # Use the last detected training run, which corresponds to the final run in Notebook 05.
            last_history = history

    return last_history, last_test_loss, last_test_acc

@st.cache_data
def load_final_classification_report(report_mtime=None):
    """Load final classification report from artifact, or parse last one from Notebook 05 outputs."""
    if os.path.exists(FINAL_REPORT_PATH):
        try:
            with open(FINAL_REPORT_PATH, "r", encoding="utf-8") as f:
                report_text = f.read().strip()
                if report_text:
                    return report_text
        except OSError:
            pass

    if not os.path.exists(NOTEBOOK_05_PATH):
        return None

    try:
        with open(NOTEBOOK_05_PATH, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    reports = []
    report_pattern = re.compile(
        r"precision\s+recall\s+f1-score\s+support[\s\S]*?weighted avg\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+[0-9]+",
        re.IGNORECASE,
    )

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            text_chunks = output.get("text", [])
            if isinstance(text_chunks, str):
                text_chunks = [text_chunks]
            merged = "".join(text_chunks)
            for match in report_pattern.findall(merged):
                reports.append(match.strip())

    if reports:
        return reports[-1]

    return None

@st.cache_data
def load_final_confusion_matrix(cm_mtime=None):
    """Load final confusion matrix saved from the last Notebook 05 section."""
    if not os.path.exists(FINAL_CONFUSION_MATRIX_PATH):
        return None
    try:
        cm = np.load(FINAL_CONFUSION_MATRIX_PATH)
        if cm.shape == (4, 4):
            return cm
    except (OSError, ValueError):
        return None
    return None

def preprocess_image(image_path, target_size=(150, 150)):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert("L")
    except (OSError, ValueError):
        return None

    # Resize using PIL then convert to numpy
    img = img.resize(target_size)
    img_array = np.array(img)
    
    # Convert grayscale to RGB (repeat channel 3 times)
    img_rgb = np.repeat(img_array[:, :, np.newaxis], 3, axis=-1)
    
    # Normalize
    img_rgb = img_rgb.astype('float32') / 255.0
    
    return img_rgb

def plot_training_history(history):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot confusion matrix"""
    cm = np.zeros((len(target_names), len(target_names)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_confusion_matrix_from_matrix(cm, target_names):
    """Plot confusion matrix from a precomputed matrix artifact."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return fig

# Page: Overview
if page == "Overview":
    st.title("🧠 Brain Tumor Detection - Project Overview")
    
    # Quick Stats Row
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", "75-82% (avg~78%)")
    with col2:
        st.metric("Classes", "4")
    with col3:
        st.metric("Input Size", "150×150")
    with col4:
        st.metric("Parameters", "19.1M")
    
    st.markdown("---")
    
    # About Project Section
    st.markdown("### 📋 Project Overview")
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("""
        **Objective:** Classify brain MRI scans using deep learning
        
        This CNN model classifies brain tumor MRI images into **4 distinct categories**:
        """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🔴 **Glioma**")
            st.caption("Highly malignant tumor originating in brain")
        with col_b:
            st.markdown("#### 🔵 **Meningioma**")
            st.caption("Usually benign tumor in membrane")
        
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("#### 🟢 **No Tumor**")
            st.caption("Healthy brain with no abnormalities")
        with col_d:
            st.markdown("#### 🟡 **Pituitary**")
            st.caption("Usually benign pituitary gland tumor")
    
    with col2:
        st.info("""
        **Dataset:**
        - Training: 1,600 images (400 per class)
        - Testing: 1,600 images (400 per class)
        - Format: Grayscale MRI scans
        - Size: 150×150 pixels each
        """)
    
    st.markdown("---")
    
    # Model Architecture
    st.markdown("### 🏗️ Model Architecture")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("**CNN Architecture Layers:**")
        st.markdown("""
        | Layer | Details |
        |-------|---------|
        | **Input** | 150×150×3 RGB |
        | **Conv Block 1** | 32 filters, 3×3 kernel |
        | **Conv Block 2** | 64 filters, 3×3 kernel |
        | **Conv Block 3** | 128 filters, 3×3 kernel |
        | **Max Pooling** | 2×2, after each block |
        | **Dense Layer 1** | 512 neurons, ReLU |
        | **Dense Layer 2** | 256 neurons, ReLU |
        | **Dropout** | 0.3 regularization |
        | **Output** | 4 neurons, Softmax |
        """)
    
    with col2:
        st.markdown("**Key Design Features:**")
        st.markdown("""
        ✅ **Regularization**
        - Dropout (0.3) prevents overfitting
        - L2 weight regularization
        
        ✅ **Data Processing**
        - Normalized to [0, 1]
        - Converted to 3-channel RGB
        - Resized to 150×150
        
        ✅ **Optimization**
        - Adam optimizer (lr=0.001)
        - Categorical crossentropy loss
        - Class weights for imbalance
        """)
    
    st.markdown("---")
    
    # Training Techniques
    st.markdown("### 🎯 Training Techniques")
    
    tab1, tab2, tab3 = st.tabs(["📸 Data Augmentation", "⚙️ Optimization", "📈 Training Config"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Augmentation Pipeline:**")
            st.markdown("""
            - 🔄 Rotation ±30°
            - ⬅️➡️ Width Shift 30%
            - ⬆️⬇️ Height Shift 30%
            - 🔍 Zoom 30%
            - 🔀 Horizontal Flip
            - 🔃 Vertical Flip
            - ✂️ Shear 15%
            """)
        
        with col2:
            st.markdown("**Purpose:**")
            st.markdown("""
            - Increases training data diversity
            - Improves model generalization
            - Prevents overfitting on limited data
            - Simulates real-world variations
            - Achieves ~60% augmentation boost
            """)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Strategy:**")
            st.markdown("""
            - **Optimizer:** Adam (lr=0.001)
            - **Loss:** Categorical Crossentropy
            - **Metrics:** Accuracy, Precision, Recall
            - **Batch Size:** 32
            - **Epochs:** 20
            """)
        
        with col2:
            st.markdown("**Handling Imbalance:**")
            st.markdown("""
            - Class weights computation
            - Balanced batch sampling
            - Per-class precision tracking
            - F1-score monitoring
            """)
    
    with tab3:
        st.markdown("**Training Configuration:**")
        st.markdown("""
        - **Early Stopping:** Yes (patience=5)
        - **Learning Rate Schedule:** ReduceLROnPlateau
        - **Validation Split:** 20%
        - **Total Trainable Params:** 19,166,020
        - **Final Validation Accuracy:** 79.19%
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 How to Use This App")
    st.markdown("""
    1. **📊 Overview** (you are here) - Learn about the project
    2. **🔍 Predict** - Upload your own MRI image for instant predictions
    3. **📈 Performance** - View detailed model metrics and training curves
    """)

# Page: Predict
elif page == "Predict":
    st.title("🔍 Make a Prediction")
    
    model = load_trained_model()
    if model is None:
        st.error("Unable to load model. Please ensure the model file exists.")
    else:
        st.write("Upload an MRI image to get a brain tumor prediction.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an MRI image (JPG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Create temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Preprocess image
                processed_img = preprocess_image(tmp_path)
                
                if processed_img is not None:
                    col1, col2 = st.columns(2)
                    
                    # Display original image
                    with col1:
                        st.subheader("Original Image")
                        original_img = np.array(Image.open(tmp_path).convert("L"))
                        st.image(original_img, use_container_width=True, clamp=True)
                    
                    # Make prediction
                    with col2:
                        st.subheader("Prediction Results")
                        
                        # Add batch dimension
                        img_batch = np.expand_dims(processed_img, axis=0)
                        
                        # Get prediction
                        prediction = model.predict(img_batch, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        
                        # Class names
                        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                        
                        # Display prediction
                        st.success(f"**Predicted Class:** {class_names[predicted_class]}")
                        st.info(f"**Confidence:** {confidence:.2%}")
                        
                        # Display all probabilities
                        st.subheader("Class Probabilities")
                        prob_data = {
                            class_names[i]: f"{prediction[0][i]:.2%}"
                            for i in range(len(class_names))
                        }
                        for cls, prob in prob_data.items():
                            st.write(f"- {cls}: {prob}")
                        
                        # Probability chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(class_names, prediction[0], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
                        ax.set_xlabel('Probability')
                        ax.set_title('Prediction Confidence by Class')
                        ax.set_xlim(0, 1)
                        st.pyplot(fig)
                else:
                    st.error("Could not process the image. Please ensure it's a valid MRI scan.")
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

# Page: Performance
elif page == "Performance":
    st.title("📊 Model Performance")
    
    # Load data
    X_test, y_test = load_test_data()
    nb05_history, nb05_test_loss, nb05_test_acc = load_notebook05_final_results()
    report_mtime = os.path.getmtime(FINAL_REPORT_PATH) if os.path.exists(FINAL_REPORT_PATH) else None
    cm_mtime = os.path.getmtime(FINAL_CONFUSION_MATRIX_PATH) if os.path.exists(FINAL_CONFUSION_MATRIX_PATH) else None
    fixed_report = load_final_classification_report(report_mtime)
    fixed_cm = load_final_confusion_matrix(cm_mtime)
    model = load_trained_model()

    if X_test is None or model is None:
        st.error("Unable to load performance data. Please ensure all data files exist.")
    else:
        # Training History
        st.header("Training Curves")
        st.write("Final training curves from Notebook 05 (data augmentation + class weights).")

        if nb05_history is not None:
            fig = plot_training_history(nb05_history)
            st.pyplot(fig)
        else:
            st.warning(
                "Final Notebook 05 training logs were not found in saved notebook outputs, "
                "so final curves cannot be shown yet."
            )
        
        st.markdown("---")
        
        # Evaluate on test set
        st.header("Test Set Evaluation")
        
        # Get predictions
        X_test_rgb = np.repeat(X_test, 3, axis=-1)
        predictions = model.predict(X_test_rgb, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Metrics: prefer final values parsed from Notebook 05 outputs.
        if nb05_test_loss is not None and nb05_test_acc is not None:
            test_loss, test_acc = nb05_test_loss, nb05_test_acc
            st.caption("Showing final test metrics captured from Notebook 05 outputs.")
        else:
            test_loss, test_acc = model.evaluate(X_test_rgb, y_test, verbose=0)
            st.caption("Notebook 05 final test metrics not found in outputs; showing live evaluation of best_model.keras.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{test_acc:.2%}")
        with col2:
            st.metric("Test Loss", f"{test_loss:.4f}")
        with col3:
            st.metric("Test Samples", len(y_test))
        
        st.markdown("---")
        
        # Confusion Matrix
        st.header("Confusion Matrix")
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        if fixed_cm is not None:
            st.caption("Showing final confusion matrix from saved last Notebook 05 output.")
            fig = plot_confusion_matrix_from_matrix(fixed_cm, class_names)
        else:
            st.caption("Final fixed confusion matrix not found; showing live confusion matrix.")
            fig = plot_confusion_matrix(y_test, y_pred, class_names)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Classification Report
        st.header("Classification Report")
        if fixed_report:
            st.caption("Showing final classification report from saved final artifact/Notebook 05 output.")
            st.code(fixed_report)
        else:
            st.caption("Final fixed report not found; showing simple per-class recall from live predictions.")
            cm_live = np.zeros((len(class_names), len(class_names)), dtype=int)
            for actual, predicted in zip(y_test, y_pred):
                cm_live[int(actual), int(predicted)] += 1

            report_lines = ["Class            Recall    Support"]
            for idx, name in enumerate(class_names):
                support = int(cm_live[idx].sum())
                recall = (cm_live[idx, idx] / support) if support else 0.0
                report_lines.append(f"{name:<15} {recall:>6.2%}   {support}")
            report = "\n".join(report_lines)
            st.code(report)
        
        st.markdown("---")
        
        # Key Insights
        st.header("⚠️ Important Notes")
        st.warning("""
        **This model is for educational purposes only:**
        - Test accuracy is ~80%, with better performance on "No Tumor" and "Pituitary" classes
        - Glioma detection (63-70%) is challenging due to similarity with other tumor types
        - Clinical deployment would require significantly higher accuracy (85%+)
        - Always consult with medical professionals for diagnosis
        """)



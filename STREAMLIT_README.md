# 🧠 Brain Tumor Detection - Streamlit Web App

## Quick Start

### Option 1: Using the Batch Script (Windows)
1. Double-click `run_app.bat` in the project folder
2. The app will automatically:
   - Activate the virtual environment
   - Install/update Streamlit
   - Launch the web app in your browser

### Option 2: Manual Terminal Launch
1. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install Streamlit (if not already installed):
   ```bash
   pip install streamlit
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. The app will open in your default browser at `http://localhost:8501`

---

## App Features

### 📋 **Overview Page**
- Project description and objectives
- Model architecture details
- Training techniques used
- Key performance metrics

### 🔍 **Predict Page**
- Upload an MRI image (JPG, PNG)
- Get instant tumor classification
- View confidence scores for all 4 tumor classes
- See probability visualization chart

### 📊 **Performance Page**
- Training curves (accuracy and loss)
- Test set evaluation metrics
- Confusion matrix visualization
- Detailed classification report
- Model limitations and important notes

---

## Supported Tumor Classes

The model classifies brain tumors into 4 categories:

1. **Glioma** - Highly malignant tumor (challenging to detect)
2. **Meningioma** - Usually benign tumor
3. **No Tumor** - Healthy brain scan
4. **Pituitary** - Usually benign tumor

---

## Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 150×150 pixels (grayscale converted to RGB)
- **Test Accuracy:** ~80%
- **Optimization:** Data augmentation, class weighting, dropout regularization
- **Framework:** TensorFlow/Keras

---

## File Structure

```
BrainTumorDetection/
├── app.py                    # Main Streamlit app
├── run_app.bat              # Quick launcher (Windows)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── results/
│   ├── best_model.keras     # Trained model
│   ├── X_test.npy           # Test images
│   ├── y_test.npy           # Test labels
│   └── training_history.npy # Training metrics
└── notebooks/               # Jupyter notebooks (training)
```

---

## Important Notes

⚠️ **Disclaimer:**
- This model is for **educational and demonstration purposes only**
- Test accuracy is ~80%, with variations by tumor type
- **NOT suitable for clinical diagnosis** - always consult medical professionals
- Clinical deployment would require 85%+ accuracy and extensive validation

---

## Troubleshooting

### App won't start
- Check that `results/best_model.keras` exists
- Ensure all `.npy` files are in the `results/` folder
- Verify Python environment is activated

### Model takes time to load first time
- Initial model loading caches in memory
- Subsequent predictions are instant

### Image upload fails
- Ensure image is JPG, PNG, or similar format
- Image should be a grayscale or RGB MRI scan
- Try another image if one fails

---

## Next Steps

To improve the app:
1. Add sample MRI images for testing
2. Deploy to cloud (Streamlit Cloud, Heroku, AWS)
3. Add ensemble predictions from multiple models
4. Implement grad-CAM visualization for model explainability
5. Add medical metadata (patient info, scan date, etc.)

---

**Built with:** Streamlit, TensorFlow/Keras, scikit-learn, OpenCV

# Live Event Detection for People's Safety Using NLP and Deep Learning

🚨 **IEEE Published Research Project - Advanced Audio-Based Threat Detection System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/IEEE-Published-red.svg)](https://ieeexplore.ieee.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.6%25-brightgreen.svg)]()

## 📖 Overviews

This project implements an intelligent **real-time threat detection system** that analyzes environmental audio to identify dangerous situations and automatically alerts emergency contacts. Using advanced NLP and Deep Learning techniques, the system achieves **96.6% accuracy** with **2.3-second response time** for critical safety applications.

### 🎯 Key Features

- **Real-time Audio Analysis**: Continuous monitoring of environmental sounds
- **Multi-Model Architecture**: LSTM, 1D-CNN, and 2D-CNN implementations
- **Instant Alert System**: SMS, Email, and WhatsApp notifications
- **High Accuracy**: 96.6% threat classification accuracy
- **Fast Response**: 2.3-second average processing time
- **Software-Only**: No additional hardware required beyond smartphones

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Live Audio    │───▶│  Preprocessing   │───▶│  Deep Learning  │
│   Recording     │    │   & Feature      │    │     Models      │
└─────────────────┘    │   Extraction     │    │ (LSTM/CNN/2D)   │
                       └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Emergency Alert │◀───│   Threat         │◀───│  Classification │
│ (SMS/Email/App) │    │  Classification  │    │   & Prediction  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔬 Research Paper

This implementation is based on our **IEEE published research**:

> **"Live Event Detection for People's Safety Using NLP and Deep Learning"**  
> *IEEE Access, Volume 12, 2024*  
> DOI: 10.1109/ACCESS.2023.3349097

**Authors**: Amrit Sen, Gayathri Rajakumaran, Miroslav Mahdal, Shola Usharani, Vezhavendhan Rajasekharan, Rajiv Vincent, **Karthikeyan Sugavanan**

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
librosa
NumPy
scikit-learn
```

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/KarthikeyanSugavanan/Live-Event-Detection-for-Peoples-Safety.git
   cd Live-Event-Detection-for-Peoples-Safety
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models** (Optional)
   ```bash
   python scripts/download_models.py
   ```

4. **Configure Alert Settings**
   ```bash
   cp config/config_template.yaml config/config.yaml
   # Edit config.yaml with your API keys and contact information
   ```

## 💻 Usage

### Basic Usage

```python
from src.live_event_detector import LiveEventDetector
from src.models import LSTMModel
from src.alert_system import AlertManager

# Initialize the system
detector = LiveEventDetector(
    model_type="LSTM",  # Options: "LSTM", "1D_CNN", "2D_CNN"
    model_path="models/lstm_model.h5",
    confidence_threshold=0.8
)

# Start real-time monitoring
detector.start_monitoring()

# Configure alert system
alert_manager = AlertManager(
    email_api_key="your_email_api_key",
    sms_api_key="your_sms_api_key",
    emergency_contacts=["emergency@example.com", "+1234567890"]
)

# Process live audio
threat_detected = detector.analyze_live_audio()
if threat_detected:
    alert_manager.send_emergency_alert(
        threat_type=detector.last_prediction,
        confidence=detector.last_confidence,
        audio_file=detector.last_recording
    )
```

### Command Line Interface

```bash
# Start real-time monitoring
python main.py --mode live --model lstm --config config/config.yaml

# Train new model
python main.py --mode train --data_path data/audio_dataset --model lstm

# Evaluate model performance
python main.py --mode evaluate --model_path models/lstm_model.h5 --test_data data/test
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Response Time |
|-------|----------|-----------|---------|----------|---------------|
| **LSTM** | **96.6%** | 95.8% | 96.2% | 96.0% | **2.3s** |
| 2D-CNN | 96.3% | 95.5% | 95.9% | 95.7% | 2.8s |
| 1D-CNN | 95.2% | 94.1% | 94.8% | 94.4% | 2.1s |

### Confusion Matrix (LSTM Model)
```
Dangerous Sounds Classification:
├── Gunshot: 98.2% accuracy
├── Glass Breaking: 96.8% accuracy  
├── Screaming: 97.1% accuracy
└── Fire Crackling: 95.4% accuracy

Normal Sounds Classification:
├── Car Horn: 96.9% accuracy
├── Dog Bark: 95.7% accuracy
├── Children Playing: 97.3% accuracy
└── Street Music: 96.1% accuracy
```

## 🎵 Audio Processing Pipeline

### Feature Extraction
- **Fast Fourier Transform (FFT)** for frequency domain analysis
- **Mel-Spectrogram** generation for deep learning optimization
- **Signal Envelope** creation for noise reduction
- **Sampling Rate Optimization** (16kHz) for real-time processing

### Data Preprocessing
```python
# Audio preprocessing pipeline
def preprocess_audio(audio_file):
    # Load and normalize audio
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    
    # Remove silence
    audio_trimmed = trim_silence(audio, threshold=20)
    
    # Generate Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_trimmed, 
        sr=sr, 
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
```

## 🧠 Deep Learning Models

### 1. LSTM (Long Short-Term Memory)
- **Architecture**: Bidirectional LSTM with skip connections
- **Use Case**: Sequential pattern recognition in audio streams
- **Performance**: 96.6% accuracy, best for temporal dependencies

### 2. 2D-CNN (2-Dimensional Convolutional Neural Network)
- **Architecture**: Multi-layer Conv2D with MaxPooling
- **Use Case**: Spectrogram pattern recognition
- **Performance**: 96.3% accuracy, excellent for frequency patterns

### 3. 1D-CNN (1-Dimensional Convolutional Neural Network)
- **Architecture**: Time-distributed 1D convolutions
- **Use Case**: Direct audio signal processing
- **Performance**: 95.2% accuracy, fastest processing time

## 📱 Alert System Integration

### Supported Platforms
- **📧 Email**: SMTP with audio attachments
- **📱 SMS**: Twilio/AWS SNS integration
- **💬 WhatsApp**: WhatsApp Business API
- **🌐 Web Dashboard**: Real-time monitoring interface

### Alert Configuration
```yaml
# config.yaml
alert_system:
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your_email@gmail.com"
    password: "your_app_password"
  
  sms:
    provider: "twilio"  # Options: twilio, aws_sns
    account_sid: "your_account_sid"
    auth_token: "your_auth_token"
  
  emergency_contacts:
    - email: "emergency@example.com"
      phone: "+1234567890"
      priority: "high"
```

## 📁 Project Structure

```
Live-Event-Detection-for-Peoples-Safety/
├── src/
│   ├── models/
│   │   ├── lstm_model.py          # LSTM implementation
│   │   ├── cnn_1d_model.py        # 1D-CNN implementation
│   │   └── cnn_2d_model.py        # 2D-CNN implementation
│   ├── preprocessing/
│   │   ├── audio_processor.py     # Audio preprocessing utilities
│   │   ├── feature_extractor.py   # Feature extraction methods
│   │   └── data_generator.py      # Custom data generators
│   ├── alert_system/
│   │   ├── email_sender.py        # Email notification system
│   │   ├── sms_sender.py          # SMS notification system
│   │   └── whatsapp_sender.py     # WhatsApp integration
│   ├── live_detector.py           # Main detection system
│   └── utils/
│       ├── audio_utils.py         # Audio utility functions
│       └── config_loader.py       # Configuration management
├── data/
│   ├── raw_audio/                 # Original audio dataset
│   ├── processed/                 # Preprocessed audio files
│   └── models/                    # Trained model files
├── notebooks/
│   ├── data_exploration.ipynb     # Dataset analysis
│   ├── model_training.ipynb       # Model training workflows
│   └── performance_analysis.ipynb # Performance evaluation
├── scripts/
│   ├── train_models.py            # Model training script
│   ├── evaluate_models.py         # Model evaluation script
│   └── deploy_system.py           # Deployment utilities
├── tests/
│   ├── test_models.py             # Model unit tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   └── test_alerts.py             # Alert system tests
├── config/
│   ├── config.yaml                # Main configuration file
│   └── model_configs/             # Model-specific configurations
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker compose setup
└── README.md                      # Project documentation
```

## 🔧 Configuration

### Model Configuration
```python
# Model hyperparameters
LSTM_CONFIG = {
    'sequence_length': 128,
    'n_features': 128,
    'lstm_units': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 35
}

CNN_2D_CONFIG = {
    'input_shape': (128, 128, 1),
    'conv_layers': [8, 16, 32, 64],
    'kernel_size': (3, 3),
    'pool_size': (2, 2),
    'dropout_rate': 0.25,
    'dense_units': 64
}
```

### Audio Processing Configuration
```python
AUDIO_CONFIG = {
    'sampling_rate': 16000,
    'duration': 1.0,  # seconds
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'threshold': 20  # for silence removal
}
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t live-event-detector .

# Run container
docker-compose up -d

# Access web interface
http://localhost:8080
```

### Cloud Deployment
```bash
# Deploy to AWS EC2
python scripts/deploy_aws.py --instance-type t3.medium

# Deploy to Google Cloud
python scripts/deploy_gcp.py --machine-type e2-standard-2

# Deploy to Azure
python scripts/deploy_azure.py --vm-size Standard_B2s
```

## 📈 Performance Optimization

### Real-time Processing Optimizations
- **Audio Buffering**: Circular buffer for continuous audio processing
- **Parallel Processing**: Multi-threading for concurrent analysis
- **Model Optimization**: TensorFlow Lite for mobile deployment
- **Memory Management**: Efficient audio chunk processing

### Scalability Features
- **Load Balancing**: Multiple model instances for high traffic
- **Database Integration**: PostgreSQL for audio logs and analytics
- **API Gateway**: RESTful API for external integrations
- **Monitoring**: Prometheus and Grafana for system monitoring

## 🔬 Research Applications

### Academic Use Cases
- **Emergency Response Systems**: Hospital and campus safety
- **Smart City Integration**: Urban threat monitoring
- **IoT Security**: Connected device threat detection
- **Healthcare Monitoring**: Patient emergency detection

### Industry Applications
- **Insurance**: Fraud detection and risk assessment
- **Security**: Corporate and residential safety systems
- **Automotive**: In-vehicle emergency detection
- **Healthcare**: Elderly care and patient monitoring

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone repository
git clone https://github.com/KarthikeyanSugavanan/Live-Event-Detection-for-Peoples-Safety.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
flake8 src/
```

## 📊 Dataset Information

### Audio Dataset Specifications
- **Total Samples**: 15,000+ audio clips
- **Audio Classes**: 13 categories
- **Dangerous Sounds**: Gunshot, Glass Breaking, Screaming, Fire Crackling
- **Normal Sounds**: Car Horn, Dog Bark, Children Playing, Street Music, etc.
- **Format**: WAV files, 16kHz sampling rate
- **Duration**: 1-second clips for training

### Data Sources
- UrbanSound8K Dataset
- Environmental Sound Classification 50
- Custom Screaming Audio Dataset
- Synthesized Emergency Sounds

## 🛡️ Security & Privacy

### Data Protection
- **Local Processing**: Audio analysis performed locally
- **Encrypted Communication**: SSL/TLS for all API communications
- **Minimal Data Storage**: Audio files automatically deleted after analysis
- **GDPR Compliance**: Privacy-first design principles

### Security Features
- **API Authentication**: JWT tokens for secure access
- **Rate Limiting**: Protection against abuse
- **Audit Logging**: Complete system activity logs
- **Emergency Override**: Manual emergency activation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Primary Author**: Karthikeyan Sugavanan
- **Email**: karthikeyan.sugavanan@example.com
- **LinkedIn**: [linkedin.com/in/karthikeyan-sugavanan](https://linkedin.com/in/karthikeyan-sugavanan)
- **IEEE Paper**: [DOI: 10.1109/ACCESS.2023.3349097](https://ieeexplore.ieee.org/)

### Research Team
- **Dr. Gayathri Rajakumaran** - VIT Chennai
- **Dr. Shola Usharani** - VIT Chennai
- **Dr. Miroslav Mahdal** - VSB-Technical University of Ostrava

## 🏆 Achievements & Recognition

- ✅ **IEEE Access Publication** (2024)
- ✅ **96.6% Classification Accuracy**
- ✅ **2.3-Second Response Time**
- ✅ **Real-time Deployment Ready**
- ✅ **Open Source Implementation**

## 🚀 Future Enhancements

### Planned Features
- [ ] **Multi-language Audio Support**
- [ ] **Edge Device Optimization** (Raspberry Pi, Mobile)
- [ ] **Advanced Neural Architectures** (Transformers, Attention Mechanisms)
- [ ] **Federated Learning** for privacy-preserving training
- [ ] **Computer Vision Integration** for multi-modal threat detection
- [ ] **Blockchain** for secure emergency response coordination

### Research Directions
- [ ] **Explainable AI** for threat detection reasoning
- [ ] **Few-shot Learning** for rapid adaptation to new threats
- [ ] **Adversarial Robustness** against audio attacks
- [ ] **Cross-domain Transfer Learning**

---

⭐ **Star this repository if you find it useful!**

📚 **For detailed technical implementation, please refer to our [IEEE paper](https://ieeexplore.ieee.org/) and [documentation](docs/).**

🔗 **Connect with us on [LinkedIn](https://linkedin.com/in/karthikeyan-sugavanan) for updates and discussions!**

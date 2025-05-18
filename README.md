# Water Potability Prediction

A machine learning application that predicts water potability based on chemical and physical properties. Built with Streamlit and scikit-learn, packaged with Docker for easy deployment.

![Application Interface](images/ui_look.png)

## Features

### Machine Learning
- Random Forest Classifier for water potability prediction
- Training accuracy: 99.89%
- Testing accuracy: 67.53%
- Input validation and error handling

### User Interface
- Interactive web interface with Streamlit
- Dark theme for better readability
- Real-time predictions
- Chemistry-themed design

### Development
- Docker containerization for easy deployment
- Comprehensive test suite (79% coverage)
- Modular code structure

## Project Structure

```
├── app.py                     # Main Streamlit application
├── config/
│   └── model_config.yaml      # Model and training configuration
├── data/
│   └── raw/
│       └── water_potability.csv  # Raw dataset
├── images/
│   ├── _.jpeg                 # Chemistry lab image for UI
│   └── ui_look.png            # UI screenshot
├── models/
│   └── potability_model.pkl   # Trained model
├── notebooks/
│   └── code.ipynb             # EDA and model development notebook
├── src/
│   ├── __init__.py
│   ├── model.py              # Model class definition
│   ├── train.py              # Model training script
│   └── utils.py              # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Model tests
│   ├── test_train.py         # Training tests
│   └── test_utils.py         # Utility tests
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── pytest.ini               # Test configuration
└── requirements.txt         # Python dependencies
```

## Installation

You can run this application either locally or using Docker.

### Local Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/water-potability-prediction.git
cd water-potability-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model with default parameters:
```bash
python src/train.py
```

This will train the model with the following performance metrics:
- Training accuracy: 99.89%
- Testing accuracy: 67.53%

Model configuration can be modified in `config/model_config.yaml`.

### Running the Application

#### Local Usage

1. Run the Streamlit app:
```bash
python -m streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter the water quality parameters and click 'Calculate' to get the prediction

#### Docker Usage

1. Build and run the container:
```bash
# Build the image
docker-compose build

# Run the container
docker run -p 8501:8501 deploiement_ml_model_using_streamlit_docker_web
```

2. Open your web browser and navigate to `http://localhost:8501`

3. To stop the container:
```bash
docker stop $(docker ps -q --filter ancestor=deploiement_ml_model_using_streamlit_docker_web)
```

## Development

### Running Tests

Run the test suite:
```bash
pytest -v
```

For coverage report:
```bash
pytest --cov=src
```

Current test coverage:
```
Name              Stmts   Miss  Cover
---------------------------------------
src/__init__.py       0      0   100%
src/model.py         50     18    64%
src/train.py         42      3    93%
src/utils.py         10      0   100%
---------------------------------------
TOTAL               102     21    79%
```

All 12 tests are passing, covering:
- Model prediction functionality
- Data loading and preprocessing
- Input validation
- Pipeline creation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
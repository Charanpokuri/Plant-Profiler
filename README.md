# Crop Information Retrieval System

A Flask-based web application that uses a trained CNN model (model.pkl) to identify crops from uploaded images and provides detailed agricultural information from a MySQL database.

## Features

- **Image Upload**: Drag-and-drop interface for crop image uploads
- **AI Crop Recognition**: Uses your trained model.pkl for crop identification
- **Detailed Crop Information**: Retrieves comprehensive data from MySQL database
- **Single Page Results**: Shows prediction and crop info on the same page
- **Responsive Design**: Beautiful CSS-only responsive UI

## Installation

1. **Install Python dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Set up your model file**
   - Place your trained `model.pkl` file in the project root directory
   - The model should be trained on the 11 crop classes: sugarcane, maize, tomato, sunflower, Cherry, jowar, wheat, cotton, rice, chilli, coconut

3. **Configure database connection**
   - Update database credentials in `app.py`:
     ```python
     DB_CONFIG = {
         'host': 'localhost',
         'user': 'your_username',
         'password': 'your_password',
         'database': 'crop_db'
     }
     ```

4. **Ensure your database has the crops table**
   - The application expects a `crops` table with crop information
   - Table should contain: name, scientific_name, description, growing_season, water_requirements, soil_type, fertilizer_needs, common_diseases, harvest_time

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
crop-information-system/
├── app.py                 # Main Flask application
├── model.pkl             # Your trained CNN model (place here)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── upload.html       # Upload and results page
└── static/
    ├── css/
    │   └── style.css     # Main stylesheet
    └── uploads/          # Uploaded images directory
```

## Model Requirements

Your `model.pkl` file should:
- Be a trained scikit-learn or joblib-compatible model
- Accept preprocessed image arrays (224x224 RGB)
- Return predictions for the 11 crop classes
- Be saved using `joblib.dump()` or `pickle.dump()`

## Supported Crops

The system supports identification of:
- sugarcane, maize, tomato, sunflower, Cherry, jowar, wheat, cotton, rice, chilli, coconut

## How It Works

1. **Upload Image**: User uploads crop image via drag-and-drop or file selection
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Prediction**: model.pkl processes the image and returns crop prediction
4. **Database Query**: Application retrieves detailed crop information from MySQL
5. **Results Display**: Shows prediction confidence and comprehensive crop details on same page

## Troubleshooting

- **Model not found**: Ensure `model.pkl` is in the project root directory
- **Database connection error**: Check your MySQL credentials in `app.py`
- **Import errors**: Make sure all dependencies are installed with `python -m pip install -r requirements.txt`

## License

This project is licensed under the MIT License.
# Visitor Gate Pass Management with Face Recognition

A sophisticated visitor management system that uses face recognition technology to streamline gate pass management. This system automates visitor identification and tracking using facial recognition, making the entry and exit process more secure and efficient.


## Features

- Real-time face recognition
- Visitor database management
- Automated gate pass generation
- Face detection and encoding
- Visitor tracking and logging

## Project Structure

```
FaceRecog_HighC/
├── data/
│   ├── captures/         # Stores captured images
│   ├── known_faces/      # Directory for storing known face images
│   ├── face_encodings.pkl    # Pickle file storing face encodings
│   └── visitor_database.json # Database for visitor information
├── main.py              # Main application file
├── diagnostics.py       # Diagnostic and utility functions
└── dlib/               # Face recognition library
```

## Requirements

- Python 3.11+
- dlib
- Additional dependencies (will be listed in requirements.txt)

## Setup

1. Clone the repository:
   
   This project uses **git submodules** (for example, `dlib`).  
   To ensure all submodules are included, run:

   ```bash
   git clone https://github.com/Abhinavt7/Visitor-gatepass-management-FaceRecog_HighC.git
   cd Visitor-gatepass-management-FaceRecog_HighC
   git submodule update --init --recursive
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set up the face database:
   
   - Can pre Add known face images to the `data/known_faces/` directory
   - Run the application to generate face encodings

## Usage

Run the main application:
```bash
python main.py
```

The system will:
- Detect faces in real-time
- Match them against the known faces database
- Generate/validate gate passes based on recognition
- Log visitor entries and exits

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

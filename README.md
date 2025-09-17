# Visitor Gate Pass Management with Face Recognition

A sophisticated visitor management system that uses face recognition technology to streamline gate pass management. This system automates visitor identification and tracking using facial recognition, making the entry and exit process more secure and efficient.

##Important Note - 

-The dlib directory was pushed as a git submodule (as indicated by the earlier warning). This means people cloning your repository will need to use git submodule update --init --recursive to get the dlib code as well.
-Make sure you have not accidentally pushed any sensitive data (the visitor database is included in the push).


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
   ```bash
   git clone https://github.com/Abhinavt7/Visitor-gatepass-management-FaceRecog_HighC.git
   ```

2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Set up the face database:
   - Add known face images to the `data/known_faces/` directory
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

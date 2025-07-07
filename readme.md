Zero-Shot Sketch-Based Image Retrieval

This project provides a web-based interface to perform zero-shot sketch-based image retrieval (ZSE-SBIR). Users can draw sketches directly on the sketchpad in their browser and see live retrieval results of matching photos from the Sketchy dataset.

🔎 Features

Live Sketchpad UI: Draw freehand sketches in the browser using a Gradio Sketchpad component.

Zero-Shot Retrieval: Retrieve top-K matching images without requiring any fine-tuning.

Caching: On first run, gallery features are computed and saved to a cache directory to speed up subsequent queries.

Example Sketch Gallery: View random example sketches and the list of available classes.

📂 Dataset

Uses the Sketchy dataset (available online).

Stores photo gallery in datasets/Sketchy/256x256/photo.

Stores sketch gallery in datasets/Sketchy/256x256/sketch/tx_*.

You must download and arrange the Sketchy dataset folder structure accordingly before running the app.

🛠 Installation & Setup

Clone the repository:

git clone [(https://github.com/rishik-ashili/ZS-SIBR](https://github.com/rishik-ashili/ZS-SIBR)
cd your-repo

Install dependencies:

pip install -r requirements.txt

Correct paths in app.py:
Open app.py and set the following constants to point to your local dataset and checkpoint paths:

CHECKPOINT_PATH = "/path/to/checkpoint/best_checkpoint.pth"
GALLERY_ROOT   = "/path/to/Sketchy/256x256/photo"
SKETCH_ROOT    = "/path/to/Sketchy/256x256/sketch/tx_000000000000_ready"

Create cache directory (optional):

mkdir cache

▶️ Running the App

python app.py

On first run, it will compute gallery features (this may take several minutes) and store them under cache/.

On subsequent runs, features will be loaded from cache for faster startup.

Access the Gradio interface at the URL printed in the console (e.g., http://127.0.0.1:7860).

🔧 Usage

Draw your sketch on the sketchpad.

Click Submit Sketch to retrieve the top-5 matching images.

Click Clear All to reset the sketchpad and results.


https://github.com/user-attachments/assets/b2a9ae04-d727-42ff-9d7d-260daf136a57



https://github.com/user-attachments/assets/dbcac828-576d-44e1-a5ee-494335804416

📁 Project Structure

├── app.py           # Main Gradio application
├── options.py       # Command-line options parser
├── model/           # ZSE-SBIR model implementation
├── datasets/        # Place Sketchy dataset here
├── cache/           # Auto-generated feature cache
├── requirements.txt # Python dependencies
└── README.md        # This file

🤝 Contributing

Feel free to submit issues or pull requests to improve the interface, add features, or fix bugs.

📜 License

This project is licensed under the MIT License. See LICENSE for details.


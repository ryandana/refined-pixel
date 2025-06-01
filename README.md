# Refined Pixel: AI-Powered Image Enhancement Suite

<div align="center">
  <img src="icon.png" alt="Refined Pixel Logo" width="128">
</div>

**Refined Pixel** is a powerful, locally-run web application that provides state-of-the-art image background removal and enhancement.  Harnessing the latest in AI, it delivers professional-grade results right on your machine, ensuring privacy and efficiency.

## ✨ Features

* **Background Removal:** Easily remove backgrounds from images with precision using multiple AI models.
* **Image Upscaling:** Enhance and increase the resolution of images with stunning clarity.
* **User-Friendly Interface:** An intuitive, Apple-inspired design for a seamless experience.
* **Offline Processing:** Process images locally without relying on external servers.
* **Progress Tracking:** Real-time feedback on processing status.
* **Optimized Performance:** CPU optimizations and smart model management for efficient processing.

## ⚙️  Tech Stack

* Flask
* Python
* AI Models: rembg, RealESRGAN, GFPGAN
* jQuery
* Tailwind CSS

## 📦  Installation

1.  **Prerequisites**

    * Python 3.8 or higher.  Check with `python3 --version` or `python --version`.
    
2.  **Setup**

    * Clone the repository.
    * Run the setup script: `python3 setup.py`.  This will:
        * Create a virtual environment.
        * Install the necessary Python dependencies.
        * Download AI models.
    * **CUDA (GPU) Support (Optional):** The setup script will detect if you have a CUDA-enabled GPU and ask if you want to install CUDA support.  Choose "y" for significantly faster processing.

3.  **Troubleshooting Installation**

    * If you encounter errors, try running the setup script as administrator.
    * Ensure your antivirus software is not blocking the installation.
    * A stable internet connection is required for the initial setup.

## 🚀  Usage

1.  **Start the Application**

    * **CPU:** Run `./run.sh` (Linux) or `run.bat` (Windows).
    * **CUDA:** Run `./run_cuda.sh` (Linux) or `run_cuda.bat` (Windows) if you installed with CUDA support.

2.  **Access the Interface**

    * The application will automatically open in your default web browser. If not, navigate to `http://127.0.0.1:\[PORT\]` (the port number will be shown in the terminal).

3.  **Processing Images**

    * **Choose Operation:** Select "Background Removal" or "Image Enhancement."
    * **Select Model:** Choose the desired AI model for the selected operation.
    * **Upload Image:** Drag and drop an image or click to select a file (PNG, JPG, WEBP).
    * **View Progress:** Track the processing progress in real-time.
    * **Download Result:** Download the processed image.

4.  **Stopping the Application**

    * Press `Ctrl + C` in the terminal to stop the server.

## 📂  Directory Structure
refined-pixel/
├── main.py           # Main application
├── setup.py          # Installation script
├── run.sh            # Run with CPU
├── run_cuda.sh       # Run with CUDA
├── install.sh        # Alternative install script
├── requirements.txt  # Python dependencies
├── templates/        # HTML templates
│   └── index.html    # Main UI
├── models/           # AI models
│   ├── upscaler/     # Upscaling models
│   └── remover/      # Background removal models
├── uploads/          # Temporary storage for uploaded images
└── processed/        # Storage for processed images

## ⚠️  Important Notes

* **Model Download:** The setup script will download the necessary AI models. Ensure you have a stable internet connection.
* **Resource Usage:** AI processing can be resource-intensive.  Close other demanding applications for best performance.
* **File Size:** Large images may take longer to process.

## 🙏  Credits

* This project utilizes the following amazing libraries and models:
    * **rembg:** Background removal.
    * **RealESRGAN:** Image upscaling.
    * **GFPGAN:** Face enhancement.
    * **Flask:** Web framework.
    * **Tailwind CSS:** CSS framework.
* A huge thanks to the developers and contributors of these projects!

## 🐛  Bug Reports and Contributions

* Please report any issues or suggestions through the project's issue tracker.
* Contributions are welcome! Feel free to fork the repository and submit pull requests.

## 📝  License

* This project is open-source and licensed under the \[License Name].

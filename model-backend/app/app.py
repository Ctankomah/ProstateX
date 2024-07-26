from flask import Flask, request, jsonify,send_file
import zipfile
import os
from zipfile import ZipFile

from modelHandler.Prostate_classification import Prostate_Classification
app = Flask(__name__)


def handleModels():
    pipeline = Prostate_Classification(
    img_classifier_path='./models/Image_Classifier.pth',
    efficientnet_b0_path='./models/EfficientNetB0-Model.pth',
    efficientnet_b1_path='./models/EfficientNetB1-Model.pth',
    resnet50_path='./models/ResNet50-Model.pth'
    )
    result,predictions = pipeline.run_pipeline(
        './prostatedata_extracted/prostatedata/t2',
        './prostatedata_extracted/prostatedata/adc',
        './prostatedata_extracted/prostatedata/bval'
    )
    print("Result:", result)
    if predictions:
        for model_name, prediction in predictions.items():
            print(f"\n{model_name} Prediction:")
            print(f"  Label: {prediction['label']}")
            print(f"  Probabilities: {prediction['probabilities']}")    
    return result, predictions


@app.route('/')
def home():
    return "Hi from here"


@app.route('/process-zip', methods=['POST'])
def process_zip():
    if 'zipfile' not in request.files:
        return jsonify({'error': 'No file provided in the request'}), 400
    
    file = request.files['zipfile']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400
    
    # Save the uploaded file
    try:
        file_path = f"prostatedata"
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    # Unzip the file
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Create a directory to extract the files to
            extract_dir = f"{os.path.splitext(file_path)[0]}_extracted"
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract all the contents into the directory
            zip_ref.extractall(extract_dir)
        
        #  delete the zip file after extraction
        os.remove(file_path)
        
    except zipfile.BadZipFile:
        return jsonify({'error': 'Invalid zip file'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to unzip file: {str(e)}'}), 500
    results,predictions=handleModels()
    print("#####")
    print(predictions)
    print(results)
    getGradcamImagesAsZip()
    return jsonify({'message': 'File uploaded and unzipped successfully','predictions':predictions,'results':results}), 200



def getGradcamImagesAsZip():
    image_dir = 'gradcam_images' 
    image_filenames = ['EfficientNetB0_gradcam.png', 'EfficientNetB1_gradcam.png', 'Original_T2W_Image_gradcam.png', 'ResNet50_gradcam.png']  
    zip_filename = 'images.zip'
    with ZipFile(zip_filename, 'w') as zip_file:
        for filename in image_filenames:
            zip_file.write(os.path.join(image_dir, filename), filename)
    return zip_filename


@app.route('/EfficientNetB0_gradcam')
def EfficientNetB0_gradcam():
    return send_file("gradcam_images/EfficientNetB0_gradcam.png", mimetype='image/png')

@app.route('/EfficientNetB1_gradcam')
def EfficientNetB1_gradcam():
    return send_file("gradcam_images/EfficientNetB1_gradcam.png", mimetype='image/png')

@app.route('/Original_T2W_Image_gradcam')
def Original_T2W_Image_gradcam():
    return send_file("gradcam_images/Original_T2W_Image_gradcam.png", mimetype='image/png')

@app.route('/ResNet50_gradcam')
def ResNet50_gradcam():
    return send_file("gradcam_images/ResNet50_gradcam.png", mimetype='image/png')





if __name__ == '__main__':
    app.run(debug=True, port=5000)



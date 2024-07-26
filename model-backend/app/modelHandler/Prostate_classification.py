import os
import cv2
import pydicom
import numpy as np
import pandas as pd

from PIL import Image
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import matplotlib.pyplot as plt
import matplotlib as matplotlib

import torch
import torchvision.transforms as transforms

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
matplotlib.use('agg')

class Prostate_Classification:
    def __init__(self, img_classifier_path, efficientnet_b0_path, efficientnet_b1_path, resnet50_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the image classifier model (ResNet101)
        self.img_classifier_model = torch.load(img_classifier_path, map_location=self.device)
        self.img_classifier_model = self.img_classifier_model.to(self.device)
        self.img_classifier_model.eval()

        # Load the three classification models
        self.efficientnet_b0 = torch.load(efficientnet_b0_path, map_location=self.device)
        self.efficientnet_b1 = torch.load(efficientnet_b1_path, map_location=self.device)
        self.resnet50 = torch.load(resnet50_path, map_location=self.device)

        self.models = {
            "EfficientNetB0": self.efficientnet_b0,
            "EfficientNetB1": self.efficientnet_b1,
            "ResNet50": self.resnet50
        }

        for model in self.models.values():
            model.to(self.device)
            model.eval()

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, t2w_folder, adc_folder, bval_folder):
        try:
            # Load DICOM images from folders
            t2w_image = self.load_dicom_series(t2w_folder)
            adc_image = self.load_dicom_series(adc_folder)
            bval_image = self.load_dicom_series(bval_folder)

            # Register ADC and b-value images to T2W image
            adc_registered = self.register_images(t2w_image, adc_image)
            bval_registered = self.register_images(t2w_image, bval_image)

            # Crop images
            t2w_cropped = self.crop_image(t2w_image)
            adc_cropped = self.crop_image(adc_registered)
            bval_cropped = self.crop_image(bval_registered)

            # Stack slices from images
            preprocessed_images = self.stack_slices_from_images(t2w_cropped, adc_cropped, bval_cropped)

            return preprocessed_images
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            return None

    def predict(self, preprocessed_images):
        # First, use the image classifier to determine if it's a prostate MRI scan
        is_prostate_mri = self.classify_prostate_mri(preprocessed_images)

        if not is_prostate_mri:
            return "Not a prostate MRI scan", None

        # If it is a prostate MRI scan, use models for clinical significance
        class_label, class_probabilities = self.classify_clinical_significance(preprocessed_images)

        return class_label, class_probabilities

    def classify_prostate_mri(self, preprocessed_images):
        for image in preprocessed_images:
            # Convert NumPy array to PIL Image before applying transforms
            image = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.img_classifier_model(image_tensor)
                probabilities = F.softmax(output, dim=1)

            # Assuming the classifier outputs a probability
            if probabilities.argmax(dim=1).item() == 1:  # Assuming class 1 means it's a prostate MRI
                return True

        return False

    def classify_clinical_significance(self, preprocessed_images):
        all_predictions = {model_name: [] for model_name in self.models.keys()}

        for image in preprocessed_images:
            image = Image.fromarray((image * 255).astype(np.uint8))
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                for model_name, model in self.models.items():
                    output = model(image_tensor)
                    probabilities = F.softmax(output, dim=1)
                    all_predictions[model_name].append(probabilities)

        final_predictions = {}
        class_labels = ["Clinically Insignificant", "Clinically Significant"]

        for model_name, predictions in all_predictions.items():
            final_prediction = torch.mean(torch.stack(predictions), dim=0)
            class_probabilities = final_prediction.squeeze().cpu().numpy() * 100

            predicted_class = torch.argmax(final_prediction).item()
            predicted_label = class_labels[predicted_class]

            formatted_probabilities = {
                "Clinically Insignificant": f"{class_probabilities[0]:.2f}%",
                "Clinically Significant": f"{class_probabilities[1]:.2f}%"
            }

            final_predictions[model_name] = {
                "label": predicted_label,
                "probabilities": formatted_probabilities
            }

        # Calculate joint prediction
        joint_prediction = torch.mean(torch.stack([
            torch.mean(torch.stack(predictions), dim=0)
            for predictions in all_predictions.values()
        ]), dim=0)

        joint_class_probabilities = joint_prediction.squeeze().cpu().numpy() * 100
        joint_predicted_class = torch.argmax(joint_prediction).item()
        joint_predicted_label = class_labels[joint_predicted_class]

        joint_formatted_probabilities = {
            "Clinically Insignificant": f"{joint_class_probabilities[0]:.2f}%",
            "Clinically Significant": f"{joint_class_probabilities[1]:.2f}%"
        }

        final_predictions["Joint"] = {
            "label": joint_predicted_label,
            "probabilities": joint_formatted_probabilities
        }

        return final_predictions

    def generate_gradcam(self, model, image_tensor, target_layer):
        model.eval()
        gradients = []
        activations = []

        def save_gradient(grad):
            gradients.append(grad)

        def forward_hook(module, input, output):
            output.register_hook(save_gradient)
            activations.append(output)
            return output

        hook = target_layer.register_forward_hook(forward_hook)
        output = model(image_tensor)
        hook.remove()

        target_class = torch.argmax(output, dim=1).item()
        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        gradients = gradients[0].cpu().data.numpy()
        activations = activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))
        gradcam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            gradcam += w * activations[0, i, :, :]

        gradcam = np.maximum(gradcam, 0)
        gradcam = cv2.resize(gradcam, (128, 128))  # Resized to match our image size
        gradcam = gradcam - gradcam.min()
        gradcam = gradcam / gradcam.max()

        return gradcam
    
    def saveGradcam_images(self,image,model_name):
            # Convert the image to a format suitable for saving
            image_img_uint8 = np.uint8(255 * image)
            # Define the directory to save the images
            save_dir = 'gradcam_images'
            os.makedirs(save_dir, exist_ok=True)
            # save images
            save_path = os.path.join(save_dir, f'{model_name}_gradcam.png')
            cv2.imwrite(save_path, image_img_uint8)

    def visualize_gradcam(self, preprocessed_images):
        # Select the middle image from preprocessed_images
        image = preprocessed_images[len(preprocessed_images) // 2]

        # Convert NumPy array to PIL Image
        image_pil = Image.fromarray((image * 255).astype(np.uint8))

        # Apply the transform
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # Create a figure with subplots for each model plus the original image
        fig, axes = plt.subplots(2, 2, figsize=(8, 5))
        axes = axes.flatten()

        # Display the original image
        original_image = image[:,:,0]  # T2W image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original T2W Image')
        axes[0].axis('off')


        self.saveGradcam_images(original_image,"Original_T2W_Image")

        # Generate and display Grad-CAM for each model
        for i, (model_name, model) in enumerate(self.models.items(), start=1):
            if 'efficientnet' in model_name.lower():
                target_layer = model.features[-1]
            elif 'resnet' in model_name.lower():
                target_layer = model.layer4[-1]
            else:
                raise ValueError(f"Unsupported model architecture: {model_name}")

            gradcam = self.generate_gradcam(model, image_tensor, target_layer)

            # Resize gradcam to match original image size
            gradcam_resized = cv2.resize(gradcam, (original_image.shape[1], original_image.shape[0]))

            # Create a color mapping of the gradcam
            gradcam_heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
            gradcam_heatmap = np.float32(gradcam_heatmap) / 255

            # Overlay the heatmap on original image
            superimposed_img = gradcam_heatmap * 0.4 + original_image[:,:,np.newaxis]
            superimposed_img = superimposed_img / np.max(superimposed_img)
            self.saveGradcam_images(superimposed_img,model_name)
            axes[i].imshow(superimposed_img)
            axes[i].set_title(f'{model_name} Grad-CAM')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


    def run_pipeline(self, t2w_folder, adc_folder, bval_folder):
        preprocessed_images = self.preprocess(t2w_folder, adc_folder, bval_folder)

        if preprocessed_images is None:
            return "Error during preprocessing", None

        is_prostate_mri = self.classify_prostate_mri(preprocessed_images)

        if not is_prostate_mri:
            return "Not a prostate MRI scan", None

        self.visualize_gradcam(preprocessed_images)
        final_predictions = self.classify_clinical_significance(preprocessed_images)
        return "Prostate MRI scan", final_predictions

    @staticmethod
    def load_dicom_series(folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError(f"The provided path is not a directory: {folder_path}")

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)

        if not dicom_names:
            raise ValueError(f"No DICOM series found in the directory: {folder_path}")

        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except RuntimeError as e:
            raise RuntimeError(f"Error reading DICOM series from {folder_path}: {str(e)}")

        return image

    @staticmethod
    def register_images(fixed_image, moving_image):
        initial_transform = sitk.CenteredTransformInitializer(
            sitk.Cast(fixed_image, moving_image.GetPixelID()),
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=1000, convergenceMinimumValue=1e-6)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(moving_image, sitk.sitkFloat32)
        )

        moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0)
        return moving_resampled

    @staticmethod
    def crop_image(image, patch_size=200, num_slices=3):
        volume = sitk.GetArrayFromImage(image)
        volume = volume.transpose((2, 1, 0))

        center_i = volume.shape[0] // 2
        center_j = volume.shape[1] // 2
        center_k = volume.shape[2] // 2

        x1 = center_j - patch_size // 2
        x2 = center_j + patch_size // 2
        y1 = center_i - patch_size // 2
        y2 = center_i + patch_size // 2

        cropped_volume = volume[y1:y2 + 1, x1:x2 + 1, center_k - num_slices:center_k + num_slices + 1]
        cropped_volume = cropped_volume.transpose((2, 1, 0))

        cropped_image = sitk.GetImageFromArray(cropped_volume)
        cropped_image.SetSpacing(image.GetSpacing())

        return cropped_image

    @staticmethod
    def stack_slices_from_images(x, y, z):
        images_for_model = []
        # Convert the cropped images to numpy arrays
        t2w_img = sitk.GetArrayFromImage(x)
        adc_img = sitk.GetArrayFromImage(y)
        bval_img = sitk.GetArrayFromImage(z)

        # Normalize the image arrays
        t2w_img = t2w_img.astype(np.float32) / t2w_img.max()
        adc_img = adc_img.astype(np.float32) / adc_img.max()
        bval_img = bval_img.astype(np.float32) / bval_img.max()

        # Stack and save the images
        for idx in range(t2w_img.shape[0]):
            image = np.stack([t2w_img[idx], bval_img[idx], adc_img[idx]], axis=-1)
            images_for_model.append(image)

        return images_for_model


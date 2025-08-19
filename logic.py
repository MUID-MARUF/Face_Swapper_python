import cv2
import insightface
from insightface.app import FaceAnalysis

def swap_faces(source_path, target_path, model_path, progress_callback=None):
    try:
        # Initialize face detector
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Load model
        swapper = insightface.model_zoo.get_model(model_path)

        # Read images
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

        if source_img is None or target_img is None:
            raise FileNotFoundError("Source or target image not found!")

        if progress_callback:
            progress_callback(20)

        # Detect faces
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)

        if len(source_faces) == 0:
            raise ValueError("No face detected in the source image!")
        if len(target_faces) == 0:
            raise ValueError("No face detected in the target image!")

        if progress_callback:
            progress_callback(50)

        # Swap first detected face
        result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

        if progress_callback:
            progress_callback(90)

        return result

    except Exception as e:
        raise RuntimeError(f"Face swapping failed: {str(e)}")

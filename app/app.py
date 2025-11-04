import os
from datetime import datetime
from pathlib import Path
import cv2
import jsonpickle
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, request
import subprocess

from facetools import FaceDetection, IdentityVerification, LivenessDetection

root = Path(os.path.abspath(__file__)).parent.absolute()
load_dotenv((root / ".env").as_posix())

data_folder = os.environ.get("DATA_FOLDER")
data_folder = (root.parent / data_folder).resolve()
resnet_name = os.environ.get("RESNET", "InceptionResnetV1_vggface2.onnx")
deeppix_name = os.environ.get("DEEPPIX", "OULU_Protocol_2_model_0_0.onnx")
facebank_name = os.environ.get("FACEBANK", "facebank.csv")

resNet_checkpoint_path = data_folder / "checkpoints" / resnet_name
facebank_path = data_folder / facebank_name
deepPix_checkpoint_path = data_folder / "checkpoints" / deeppix_name

faceDetector = FaceDetection()
identityChecker = IdentityVerification(
    checkpoint_path=resNet_checkpoint_path.as_posix(),
    facebank_path=facebank_path.as_posix(),
)
livenessDetector = LivenessDetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

app = Flask(__name__)

@app.route("/main", methods=["POST"])
def main():
    r = request
    nparr = np.frombuffer(r.data, np.uint8)

    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        response = {"message": "Invalid image data."}
        return Response(jsonpickle.encode(response), status=400, mimetype="application/json")

    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {
            "message": "No faces detected.",
            "faces": []
        }
        status_code = 200
    else:
        results = []
        for idx, face_arr in enumerate(faces):
            min_sim_score, mean_sim_score = identityChecker(face_arr)
            liveness_score = livenessDetector(face_arr)

            results.append({
                "id": idx,
                "min_sim_score": float(min_sim_score),
                "mean_sim_score": float(mean_sim_score),
                "liveness_score": float(liveness_score),
            })

            # spoof detection log
            if float(liveness_score) < 0.5:
                log_path = data_folder / "spoof_log.csv"
                with open(log_path, "a") as f:
                    f.write(f"{datetime.now()},{mean_sim_score},{liveness_score}\n")

        response = {
            "message": "Faces processed successfully.",
            "faces": results,
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=status_code, mimetype="application/json")


@app.route("/identity", methods=["POST"])
def identity():
    r = request
    nparr = np.frombuffer(r.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {"message": "No faces detected.", "min_sim_score": None, "mean_sim_score": None}
        status_code = 500
    else:
        face_arr = faces[0]
        min_sim_score, mean_sim_score = identityChecker(face_arr)
        response = {
            "message": "Face identified.",
            "min_sim_score": float(min_sim_score),
            "mean_sim_score": float(mean_sim_score),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=status_code, mimetype="application/json")


@app.route("/liveness", methods=["POST"])
def liveness():
    r = request
    nparr = np.frombuffer(r.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(frame)

    if not len(faces):
        response = {"message": "No faces detected.", "liveness_score": None}
        status_code = 500
    else:
        face_arr = faces[0]
        liveness_score = livenessDetector(face_arr)
        response = {
            "message": "Liveness detected.",
            "liveness_score": float(liveness_score),
        }
        status_code = 200

    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=status_code, mimetype="application/json")

@app.route("/start_stream", methods=["POST"])
def start_stream():
    try:
        subprocess.Popen(["python", "app/stream_client.py"])
        return {"message": "Webcam streaming started."}, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/healthz", methods=["GET"])
def health():
    """Health check endpoint for Render / Docker."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

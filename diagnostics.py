# diagnostics.py
import sys, os
import dlib, face_recognition

print("python exe:", sys.executable)
print("python version:", sys.version.splitlines()[0])
print("pip:", os.popen(f'"{sys.executable}" -m pip --version').read().strip())
print("dlib file:", getattr(dlib, "__file__", None))
print("dlib.DLIB_USE_CUDA:", getattr(dlib, "DLIB_USE_CUDA", None))
print("has dlib.cuda:", hasattr(dlib, "cuda"))
if hasattr(dlib, "cuda"):
    try:
        print("dlib.cuda.get_num_devices():", dlib.cuda.get_num_devices())
    except Exception as e:
        print("dlib.cuda.get_num_devices() error:", e)
print("face_recognition file:", getattr(face_recognition, "__file__", None))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

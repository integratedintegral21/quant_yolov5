import onnxruntime as ort
import numpy as np
import cv2

def main():
    device = 'cuda'
    ort_sess = ort.InferenceSession('yolov5s_qat_w8a8_cones/weights/model_optimized.onnx')
    # ort_sess = ort.InferenceSession('yolov5s_cones_multiscale/weights/best.onnx')
    inp = cv2.imread('data/images/cones.png')
    inp = cv2.resize(inp, (640, 640))
    inp = inp / 255.
    inp = np.reshape(inp, (1, inp.shape[0], inp.shape[1], inp.shape[2]))
    inp = np.transpose(inp, (0, 3, 1, 2))
    inp = np.float32(inp)
    out = ort_sess.run(None, {ort_sess.get_inputs()[0].name: inp})
    print(out[0])


if __name__ == "__main__":
    main()

import onnx
from onnxconverter_common import float16


def main():
    model = onnx.load("yolov5s_fp32.weights")
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, "yolov5s_fp16.weights")


if __name__ == "__main__":
    main()

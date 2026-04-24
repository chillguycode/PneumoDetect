import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
from pathlib import Path

def generate_dummy_model(output_path: Path):
    if output_path.exists():
        print(f"Dummy model already exists at {output_path}")
        return

    print(f"Generating dummy ONNX model at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W = numpy_helper.from_array(np.zeros((150528, 2), dtype=np.float32), name='W')
    B = numpy_helper.from_array(np.zeros(2, dtype=np.float32), name='B')

    graph = helper.make_graph(
        [
            helper.make_node('Flatten', inputs=['input'], outputs=['flat'], axis=1),
            helper.make_node('Gemm', inputs=['flat', 'W', 'B'], outputs=['output']),
        ],
        'dummy',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])],
        initializer=[W, B]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print("Dummy model generated successfully")


if __name__ == "__main__":
    MODEL_DIR = Path(__file__).parent.parent.parent/ "saved_models_test"
    generate_dummy_model(MODEL_DIR/"dummy_model.onnx")

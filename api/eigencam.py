from PIL import Image 
import onnx
import onnxruntime as ort
import numpy as np
import io
import base64

EIGENCAM_TARGET_NODE = '/features/features.7/features.7.2/Mul_output_0'

def create_eigencam_session(model_path:str, target_node:str):
    model = onnx.load(model_path)
    target_layer = onnx.helper.make_tensor_value_info(
        target_node,
        onnx.TensorProto.FLOAT,
        None
    )
    model.graph.output.append(target_layer)
    session = ort.InferenceSession(model.SerializeToString())

    return session


def get_feature_maps(session, tensor:np.ndarray, target_node:str):
    outputs = session.run(
        [target_node],
        {"input": tensor}
    )
    
    return outputs


def compute_eigencam(feature_map):
    features = np.squeeze(feature_map, axis=0)
    C,H,W = features.shape
    features = features.reshape(C,-1)
    _,_,Vt = np.linalg.svd(features,full_matrices=False)
    activation_map = Vt[0].reshape(H,W)

    if activation_map.mean() < 0:
        activation_map = -activation_map
    return activation_map


def upsample_heatmap(activation_map: np.ndarray, target_size:tuple[int,int]) -> np.ndarray :
    range_val = activation_map.max() - activation_map.min()
    if range_val < 1e-8:
        normalised_map = np.zeros_like(activation_map)
    else:
        normalised_map = (activation_map - activation_map.min()) / range_val
    pil_image = Image.fromarray(normalised_map)
    pil_image = pil_image.resize(target_size, Image.Resampling.BILINEAR)
    heatmap = np.array(pil_image)

    return heatmap


def _jet(t):
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b, np.ones_like(t)], axis=-1)


def render_heatmap(heatmap: np.ndarray):
    rgba = _jet(heatmap)
    rgba_uint8 = (rgba*255).astype(np.uint8)
    heatmap_img = Image.fromarray(rgba_uint8)

    buffer = io.BytesIO()
    heatmap_img.save(buffer,format="PNG")
    
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return encoded

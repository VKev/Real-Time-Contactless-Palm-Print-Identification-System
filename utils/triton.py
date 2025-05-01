import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput

dtype_map = {
    np.dtype('float32'): "FP32",
    np.dtype('float64'): "FP64",
    np.dtype('int32'):   "INT32",
    np.dtype('int64'):   "INT64",
    np.dtype('uint8'):   "UINT8",
    np.dtype('bool'):    "BOOL",
}

class TritonClient:
    """
    Simple gRPC client for Triton Inference Server.
    Usage:
        client = TritonClient(url="localhost:8001")
        inputs = {"INPUT__0": np.ndarray}
        # outputs can be omitted to fetch all model outputs
        result = client.infer("feature_extraction", inputs)
    """
    def __init__(self, url: str = "localhost:8001", verbose: bool = False):
        self.client = InferenceServerClient(url=url, verbose=verbose)

    def infer(
        self,
        model_name: str,
        inputs: dict,
        outputs: list = None,
        model_version: str = "",
        timeout: float = None
    ) -> dict:
        if outputs is None:
            metadata = self.client.get_model_metadata(
                model_name=model_name,
                model_version=model_version
            )
            outputs = [out.name for out in metadata.outputs]

        infer_inputs = []
        for name, array in inputs.items():
            arr = np.ascontiguousarray(array)
            np_dtype = arr.dtype
            triton_dtype = dtype_map.get(np_dtype)
            if triton_dtype is None:
                raise ValueError(f"Unsupported numpy dtype: {np_dtype} for input '{name}'")
            infer_in = InferInput(name, arr.shape, triton_dtype)
            infer_in.set_data_from_numpy(arr)
            infer_inputs.append(infer_in)

        infer_outputs = [InferRequestedOutput(out_name) for out_name in outputs]

        response = self.client.infer(
            model_name=model_name,
            inputs=infer_inputs,
            outputs=infer_outputs,
            model_version=model_version,
            timeout=timeout,
        )

        results = {}
        for out_name in outputs:
            results[out_name] = response.as_numpy(out_name)
        return results

if __name__ == "__main__":
    client = TritonClient("localhost:8001")
    dummy_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
    result = client.infer(
        model_name="feature_extraction",
        inputs={"INPUT__0": dummy_input}
    )
    print("Inference result:", result)

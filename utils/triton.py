import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from functools import partial
dtype_map = {
    np.dtype('float32'): "FP32",
    np.dtype('float64'): "FP64",
    np.dtype('int32'):   "INT32",
    np.dtype('int64'):   "INT64",
    np.dtype('uint8'):   "UINT8",
    np.dtype('bool'):    "BOOL",
}

class TritonClient:
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
        return self._do_infer(model_name, inputs, outputs, model_version, timeout, async_mode=False)

    def infer_async(
        self,
        model_name: str,
        inputs: dict,
        outputs: list = None,
        model_version: str = "",
        timeout: float = None,
        callback=None,
        user_data=None
    ):
        return self._do_infer(model_name, inputs, outputs, model_version, timeout, async_mode=True,
                              callback=callback, user_data=user_data)

    def _do_infer(
        self,
        model_name: str,
        inputs: dict,
        outputs: list,
        model_version: str,
        timeout: float,
        async_mode: bool,
        callback=None,
        user_data=None
    ):
        if outputs is None:
            meta = self.client.get_model_metadata(model_name=model_name,
                                                  model_version=model_version)
            outputs = [o.name for o in meta.outputs]

        infer_inputs = []
        for name, array in inputs.items():
            arr = np.ascontiguousarray(array)
            triton_dtype = dtype_map.get(arr.dtype)
            if triton_dtype is None:
                raise ValueError(f"Unsupported numpy dtype: {arr.dtype} for input '{name}'")
            infer_in = InferInput(name, arr.shape, triton_dtype)
            infer_in.set_data_from_numpy(arr)
            infer_inputs.append(infer_in)

        infer_outputs = [InferRequestedOutput(o) for o in outputs]

        if async_mode:
            cb = callback or (lambda ud, result, error: None)
            return self.client.async_infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                callback=partial(cb, user_data),
                model_version=model_version,
                timeout=timeout
            )
        else:
            resp = self.client.infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                model_version=model_version,
                timeout=timeout
            )
            return {o: resp.as_numpy(o) for o in outputs}


if __name__ == "__main__":
    client = TritonClient("localhost:8001")
    dummy = np.random.random((1, 3, 224, 224)).astype(np.float32)
    out = client.infer(model_name="feature_extraction", inputs={"INPUT__0": dummy})
    print("Sync inference result shapes:", {k: v.shape for k, v in out.items()})
    
    import queue, time
    doneQ = queue.Queue()
    def cb(user_data, result, error):
        if error:
            print("Async error:", error)
        else:
            print("Async result shape:", result.as_numpy("OUTPUT__0").shape)

    fut = client.infer_async("feature_extraction", {"INPUT__0": dummy}, callback=cb, user_data=None)
    time.sleep(1)

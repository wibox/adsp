import os
import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from onnxruntime import InferenceSession, SessionOptions, ExecutionMode

from serve.common.handlers import BaseHandler


class FireNetHandler(BaseHandler):
    """Flood detection model using satellite images to map flooded areas
    """

    def __init__(self, context) -> None:
        super().__init__(context)

    def initialize_session(self) -> InferenceSession:
        model_path = os.path.join(self.context.package_dir, "concat_unet-v1.0.onnx")
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = 8
        sess_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        return InferenceSession(model_path)

    def setup(self) -> None:
        """Setup hook, should be implemented even if no-op.
        Provides a useful lifecycle function to initialize stuff like labels and model specific stuff.

        :raises NotImplementedError: implement in a subclass
        """
        pass

    def preprocess(self, data: Optional[Dict], image: np.ndarray,
                   **params) -> List[Any]:
        """
        Preprocessing step, where the generic input is transformed into tensors or other inputs.
        The result is in list form, where each element correspond to the respective input.
        :param data: generic dictionary containing input data
        :type data: Optional[Dict]
        :param files: optional list of files, already available locally
        :type files: Optional[List]
        :param params: additional key-value parameters, if needed
        :type params: dict
        :return: list of preprocessed inputs
        :rtype: list
        """
        # Make image brighter -> network was trained on brighter images (since RGB bands [3, 2, 1] are brightened for
        # better visualization
        image = (image).astype(np.float32)
        image = (image - 0.5) / 0.5
        return [image]

    def predict(self, data: Optional[Dict], image: np.ndarray, **params) -> Tuple[Any, List[np.ndarray]]:
        """Main method, implementing the prediction steps using the ONNX Runtime.
        Preprocessed data is provided to the inference session, then a postprocessing step makes it
        ready to be returned as final output.

        :param data: generic input dictionary with user-defined data, can be null
        :type data: Optional[Dict]
        :param uris: possibily empty list of files to be loaded and used in inference
        :type uris: Optional[List]
        :raises ValueError: whenever the session has not been initialized properly
        :return: processed model output, containing JSON-compatible data and a list of stored files
        :rtype: Tuple[Dict, List[str]]
        """
        if self.session is None:
            raise ValueError("ONNX session not initialized, did you initialize the session?")
        # prepare inputs
        input_data = self.preprocess(data, image, **params)
        input_names = [x.name for x in self.session.get_inputs()]
        input_dict = dict(zip(input_names, input_data))
        # inference step
        outputs = self.session.run(None, input_feed=input_dict, run_options=self.options)
        # process output and return
        return self.postprocess(outputs=outputs, **params)


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def postprocess(self, outputs: Any, **params) -> Tuple[Dict,List]:
        """Postprocessing step where the model output can be further elaborated and transformed.

        :param outputs: raw output of the ONNX model
        :type outputs: Any
        :raises NotImplementedError: implement this function in a subclass
        :return: tuple containing data generated and optional stored files
        :rtype: Tuple[Dict,List]
        """
        bin_out, regr_out = outputs
        bin_out = (np.argmax(bin_out, axis=1) * 255).astype(np.uint8)
        regr_out = regr_out.squeeze(axis=1)
        regr_out = (np.clip(regr_out, 0, 4) / 4.0) * 255
        regr_out = regr_out.astype(np.uint8)
        return bin_out, regr_out


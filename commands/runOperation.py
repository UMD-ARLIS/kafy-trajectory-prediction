# commands/runOperation.py

import os
import pickle
import importlib.util
import logging

import h3
import torch

from coreComponents.transformers_plugin import TransformersPlugin
from coreComponents.tokenization import TrajectoryTokenizer
from coreComponents.trajectory_operations_plugin import TrajectoryOperationsPlugin
from coreComponents.partitioning import PartitioningModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_operation(path: str, operation_name: str, partitioning_input_path: str, trajplug_input: dict):
    """
    Executes an operation on input trajectories.

    Args:
        operation_name (str): Name of the registered operation.
        partitioning_input_path (str): Path to the trajectories input file.
        trajplug_input (dict): Additional parameters passed to the operation script.

    Returns:
        list: One result per trajectory. Spatial results are lists of (lat, lon).
    """
    # ----------------------------------------------------------
    # 1. Select the model for the region covering the input trajectories
    # ----------------------------------------------------------
    PT = PartitioningModule(project_path=path)
    model_path, trajectories = PT.select_model(operation_name, partitioning_input_path)
    logger.info(f"Selected model: {model_path}")

    # ----------------------------------------------------------
    # 2. Load tokenizer + transformer name from the model metadata
    # ----------------------------------------------------------
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.pt', '')
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    tokenizer = metadata["tokenizer"]
    transformer_name = metadata["transformer"]

    # ----------------------------------------------------------
    # 3. Rebuild the model and load the trained weights
    # ----------------------------------------------------------
    build_model_fn = TransformersPlugin(path).load_model_builder(transformer_name)
    model = build_model_fn(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # ----------------------------------------------------------
    # 4. Load the operation logic (function named after the operation)
    # ----------------------------------------------------------
    operation_data = TrajectoryOperationsPlugin(path).get(operation_name)
    if operation_data is None:
        raise ValueError(f"Operation '{operation_name}' is not registered.")
    spec = importlib.util.spec_from_file_location(operation_name, operation_data["script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logic = getattr(module, operation_name)

    # ----------------------------------------------------------
    # 5. Tokenize -> run operation -> detokenize, per trajectory
    # ----------------------------------------------------------
    final_results = []
    for traj in trajectories:
        gps_string = ",".join(f"{p.y} {p.x}" for p in traj)
        h3_string = TrajectoryTokenizer.gps_string_to_h3(gps_string, tokenizer.resolution)
        input_ids = torch.tensor([tokenizer.encode(h3_string)])

        with torch.no_grad():
            result = logic(model, input_ids, **trajplug_input)

        if torch.is_tensor(result):
            result = result[0].tolist()

        if tokenizer.is_spatial(result):
            # Drop special tokens (pad, decoder start, eos, ...) and map cells back to GPS
            tokens = [t for t in tokenizer.decode(result).split() if h3.is_valid_cell(t)]
            result = TrajectoryTokenizer.h3_to_gps(" ".join(tokens))

        logger.info(f"Result: {result}")
        final_results.append(result)

    return final_results

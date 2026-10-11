# coreComponents/trajectory_operations_plugin.py

import os
import json
import logging
import inspect
import importlib.util

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TrajectoryOperationsPlugin:

    def __init__(self, base_path):
        self.base = base_path
        self.operations_dir = os.path.join(base_path, "operations")
        os.makedirs(self.operations_dir, exist_ok=True)
        self.path = os.path.join(self.operations_dir, "operations.json")

        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f, indent=4)

    # ----------------------------------------------------------
    # Load registry
    # ----------------------------------------------------------
    def load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    # ----------------------------------------------------------
    # Function 1: Check if operation already has this transformer
    # ----------------------------------------------------------
    def has_transformer(self, op_name, transformer_name):
        """Check if operation already has this transformer registered."""
        registry = self.load()
        if op_name not in registry:
            return False
        existing_transformers = registry[op_name].get("transformers", [])
        return transformer_name in existing_transformers

    def validate_operation_script(self,file_path, operation_name):
        """
        Validates:
            - File loads successfully
            - Function with same name as operation exists in the file
            - Function accepts at least two parameters (model and tokenized trajectory)
            - Additional parameters must be handled via *args or **kwargs
        """
        # Load the module
        spec = importlib.util.spec_from_file_location("plugin_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if a function with the operation name exists
        if not hasattr(module, operation_name):
            raise ValueError(
                f"Function '{operation_name}' not found in '{file_path}'."
            )
        
        # Get the attribute
        func = getattr(module, operation_name)
        
        # Check if it's a function (not a class)
        if not inspect.isfunction(func):
            raise ValueError(f"'{operation_name}' is not a function in '{file_path}'. It is a {type(func).__name__}.")
        
        # Check function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        
        # Check if function accepts at least 2 positional parameters
        if len(params) < 2:
            raise ValueError(
                f"Function '{operation_name}' must accept at least 2 parameters "
                f"(model and tokenized trajectory). Found only {len(params)} parameters."
            )
        
        # Check if first two parameters are positional (not keyword-only)
        for i, param in enumerate(params[:2]):
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise ValueError(
                    f"Parameter {i+1} in function '{operation_name}' must be a positional parameter, "
                    f"but it's keyword-only (name: {param.name})."
                )
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise ValueError(
                    f"Parameter {i+1} in function '{operation_name}' must be a regular positional parameter, "
                    f"not *args (variadic)."
                )
        
        # Check if function accepts **kwargs for additional parameters (optional but recommended)
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if not has_kwargs:
            logging.warning(f"⚠️  Warning: Function '{operation_name}' doesn't accept **kwargs. "
                f"It won't be able to receive additional parameters from the system.")
        
        # Optional: Check if the function is callable
        if not callable(func):
            raise ValueError(f"'{operation_name}' is not callable.")
        
        # Get parameter names for logging
        param_names = [p.name for p in params]
        logging.info(f"Function '{operation_name}' validated. Parameters: {param_names}")
        
        return True
    
    # ----------
    # ----------------------------------------------------------
    # Function 2: Register operation AFTER successful training
    # ----------------------------------------------------------
    def register(self, op_name, script_path, transformer_name, model_path):
        """
        Register an operation with a transformer.
        Call this ONLY AFTER training is successful.
        """
        registry = self.load()
        # I already validated this script file in my add_operation command flow
        dest_path = os.path.join(self.operations_dir, f"{op_name}.py")

        # Copy operation script into operations/
        # (skip if it is already there: opening dest for writing would truncate it)
        if os.path.abspath(script_path) != os.path.abspath(dest_path):
            with open(script_path, "r") as src:
                with open(dest_path, "w") as dst:
                    dst.write(src.read())
                
        if op_name not in registry:
            # Create new operation entry
            registry[op_name] = {
                "transformers": [transformer_name],
                "script": dest_path
            }
            logging.info(f"New operation '{op_name}' registered with transformer '{transformer_name}'")
        else:
            # Operation already exists, add transformer to the list
            existing_transformers = registry[op_name].get("transformers", [])
            
            if transformer_name not in existing_transformers:
                existing_transformers.append(transformer_name)
                registry[op_name]["transformers"] = existing_transformers
                logging.info(f"Added transformer '{transformer_name}' to existing operation '{op_name}'")
            

        
        with open(self.path, "w") as f:
            json.dump(registry, f, indent=4)

        
    # ----------------------------------------------------------
    # Lookup operation
    # ----------------------------------------------------------
    def get(self, op_name):
        registry = self.load()
        return registry.get(op_name, None)

    def exists(self, op_name):
        registry = self.load()
        return op_name in registry
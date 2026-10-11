# components/transformers_plugin.py

import os
import importlib.util
import inspect
import json
from transformers import PreTrainedModel


class TransformersPlugin:
    def __init__(self, base_path):
        self.base_path = base_path
        self.transformers_dir = os.path.join(base_path, "transformers")
        self.registry_file = os.path.join(self.transformers_dir, "transformers_registry.json")

        os.makedirs(self.transformers_dir, exist_ok=True)

        if not os.path.exists(self.registry_file):
            with open(self.registry_file, "w") as f:
                json.dump({}, f)

    # -------------------------------------------------------
    # Registry Helpers
    # -------------------------------------------------------

    def load_registry(self):
        with open(self.registry_file, "r") as f:
            return json.load(f)

    def save_registry(self, registry):
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=4)

    # -------------------------------------------------------
    # Validation
    # -------------------------------------------------------

    @staticmethod
    def validate_transformer_plugin(file_path, transformer_name):
        """
        Validates:
            - File loads successfully
            - Class exists
            - Class inherits PreTrainedModel
        """
        spec = importlib.util.spec_from_file_location("plugin_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, transformer_name):
            raise ValueError(
                f"Class '{transformer_name}' not found in '{file_path}'."
            )

        cls = getattr(module, transformer_name)

        if not inspect.isclass(cls):
            raise ValueError(f"'{transformer_name}' is not a class.")

        if not issubclass(cls, PreTrainedModel):
            raise TypeError(
                f"Transformer '{transformer_name}' must inherit from PreTrainedModel."
            )

        return True

    # -------------------------------------------------------
    # Command: AddTransformer TransformerName ImplementationFile
    # -------------------------------------------------------

    def add_transformer(self, name, implementation_path):
        """
        Adds a transformer plugin:
            AddTransformer TransformerName ImplementationFile
        """

        # Validate BEFORE copying
        self.validate_transformer_plugin(implementation_path, name)

        registry = self.load_registry()
        dest_path = os.path.join(self.transformers_dir, f"{name}.py")

        # Copy implementation file into transformers/
        # (skip if it is already there: opening dest for writing would truncate it)
        if os.path.abspath(implementation_path) != os.path.abspath(dest_path):
            with open(implementation_path, "r") as src:
                with open(dest_path, "w") as dst:
                    dst.write(src.read())

        # Register
        registry[name] = dest_path
        self.save_registry(registry)

        print(f"[KAFY] Transformer '{name}' successfully added.")

    # -------------------------------------------------------
    # Loading the Transformer Class
    # -------------------------------------------------------

    # def load_transformer_class(self, name):
    #     registry = self.load_registry()

    #     if name not in registry:
    #         raise ValueError(f"Transformer '{name}' not found in registry.")

    #     file_path = registry[name]

    #     spec = importlib.util.spec_from_file_location(name, file_path)
    #     module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(module)

    #     if not hasattr(module, name):
    #         raise ValueError(f"File {file_path} must define class '{name}'.")

    #     return getattr(module, name)
    def load_model_builder(self, name):
        """
        Returns the build_model(tokenizer) function from the plugin file.
        """

        registry = self.load_registry()

        if name not in registry:
            raise ValueError(f"Transformer '{name}' not found in registry.")

        file_path = registry[name]

        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "build_model"):
            raise ValueError(
                f"Transformer plugin '{name}' must define a function `build_model(tokenizer)`."
            )

        return module.build_model
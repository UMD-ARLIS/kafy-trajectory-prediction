"""
next_point.py  -  KAFY operation script for next-point prediction.

KAFY rules this file must follow:
  * the file name must contain "next" (KAFY infers the task type from it)
  * the function name must equal the OperationName you use in AddOperation / RunOperation
  * first two parameters: model, input_ids ; everything else arrives via **kwargs
"""


def NextPoint(model, input_ids, **kwargs):
    # The model was trained to output exactly one H3 cell (the next point),
    # so generate a single new token. KAFY turns it back into (lat, lon).
    num_points = int(kwargs.get("num_points", 1))
    return model.generate(input_ids, max_new_tokens=num_points, do_sample=False)
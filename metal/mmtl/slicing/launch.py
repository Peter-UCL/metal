"""
Creates slice/baseline/ablation model, trains, and evaluates on
corresponding slice prediction labelsets.

python compare_to_baseline.py --seed 1 --tasks RTE --slice_dict '{"RTE": ["dash_semicolon", "more_people", "BASE"]}' --model_type naive --n_epochs 1
"""

import argparse
import copy
import json
from pprint import pprint

import numpy as np

from metal.mmtl.glue.glue_tasks import create_glue_tasks_payloads, task_defaults
from metal.mmtl.metal_model import MetalModel, model_defaults
from metal.mmtl.slicing.slice_model import SliceModel, SliceRepModel
from metal.mmtl.slicing.tasks import convert_to_slicing_tasks
from metal.mmtl.trainer import MultitaskTrainer, trainer_defaults
from metal.utils import add_flags_from_config, recursive_merge_dicts

# Overwrite defaults
task_defaults["attention"] = False
model_defaults["verbose"] = False
model_defaults["delete_heads"] = True  # mainly load the base representation weights

# by default, log last epoch (not best)
trainer_defaults["checkpoint"] = True
trainer_defaults["checkpoint_config"]["checkpoint_best"] = False
trainer_defaults["writer"] = "tensorboard"
trainer_defaults["metrics_config"][
    "test_split"
] = "valid"  # for GLUE, don't have real test set

# Model configs
model_configs = {
    "naive": {"model_class": MetalModel, "active_slice_heads": None},
    "hard_param": {
        "model_class": MetalModel,
        "active_slice_heads": {"pred": True, "ind": False},
    },
    "soft_param": {
        "model_class": SliceModel,
        "active_slice_heads": {"pred": True, "ind": True},
    },
    "soft_param_rep": {
        "model_class": SliceRepModel,
        "active_slice_heads": {"pred": False, "ind": True},
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch slicing models/baselines on GLUE tasks", add_help=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=np.random.randint(1e6),
        help="A single seed to use for trainer, model, and task configs",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(model_configs.keys()),
        help="Model to run and evaluate",
    )
    parser.add_argument(
        "--validate_on_slices",
        type=bool,
        default=False,
        help="Whether to map eval main head on validation set during training",
    )

    parser = add_flags_from_config(parser, trainer_defaults)
    parser = add_flags_from_config(parser, model_defaults)
    parser = add_flags_from_config(parser, task_defaults)
    args = parser.parse_args()

    # Extract flags into their respective config files
    trainer_config = recursive_merge_dicts(
        trainer_defaults, vars(args), misses="ignore"
    )
    model_config = recursive_merge_dicts(model_defaults, vars(args), misses="ignore")
    task_config = recursive_merge_dicts(task_defaults, vars(args), misses="ignore")
    args = parser.parse_args()

    task_names = args.tasks.split(",")
    assert len(task_names) == 1
    base_task_name = task_names[0]

    # Default name for log directory to task names
    if args.run_name is None:
        run_name = f"{args.model_type}_{args.tasks}"
        trainer_config["writer_config"]["run_name"] = run_name

    # Get model configs
    config = model_configs[args.model_type]
    active_slice_heads = config["active_slice_heads"]
    model_class = config["model_class"]

    # Create tasks and payloads
    slice_dict = json.loads(args.slice_dict) if args.slice_dict else {}
    if active_slice_heads:
        task_config.update({"slice_dict": slice_dict})
        task_config["active_slice_heads"] = active_slice_heads
    else:
        task_config.update({"slice_dict": None})
    tasks, payloads = create_glue_tasks_payloads(task_names, **task_config)

    # Create evaluation payload with test_slices -> primary task head
    task_config.update({"slice_dict": slice_dict})
    task_config["active_slice_heads"] = {
        # turn pred labelsets on, and use model's value for ind head
        "pred": True,
        "ind": active_slice_heads.get("ind", False),
    }
    slice_tasks, slice_payloads = create_glue_tasks_payloads(task_names, **task_config)
    eval_payload = slice_payloads[1]  # eval on dev scores
    pred_labelsets = [
        labelset
        for labelset in eval_payload.labels_to_tasks.keys()
        if "pred" in labelset or "_gold" in labelset
    ]
    # Only eval "pred" labelsets on main task head -- continue eval of inds on ind-heads
    eval_payload.remap_labelsets(
        {pred_labelset: base_task_name for pred_labelset in pred_labelsets}
    )

    if args.validate_on_slices:
        print("Will compute validation scores for slices based on main head.")
        payloads[1] = eval_payload

    if active_slice_heads:
        tasks = convert_to_slicing_tasks(tasks)

    # Initialize and train model
    model = model_class(tasks, **model_config)

    trainer = MultitaskTrainer(**trainer_config)
    trainer.train_model(model, payloads)

    # Evaluate trained model on slices
    model.eval()
    slice_metrics = model.score(eval_payload)
    pprint(slice_metrics)
    if trainer.writer:
        trainer.writer.write_metrics(slice_metrics, "slice_metrics.json")

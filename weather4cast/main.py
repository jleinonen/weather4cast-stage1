import argparse

import datasets
import models


def generate_predictions(
    submission_dir,
    comp_dir="w4c-core-stage-1",
    regions=None,
):
    if regions is None:
        if "core" in comp_dir:
            regions = ["R1", "R2", "R3"]
        else:
            regions = ["R4", "R5", "R6"]
    
    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="test",
        augment=False,
        shuffle=False
    )

    comb_model = models.combined_model_with_weights(batch_gen_valid)
    
    datasets.generate_submission(
        comb_model,
        submission_dir,
        regions=regions,
        comp_dir=comp_dir
    )


def evaluate(
    comp_dir="w4c-core-stage-1",
    regions=None,
    dataset="CTTH",
    variable="temperature",
    batch_size=32,
    model_type="resgru",
    weight_fn=None
):
    if regions is None:
        if "core" in comp_dir:
            regions = ["R1", "R2", "R3"]
        else:
            regions = ["R4", "R5", "R6"]

    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="validation",
        augment=False,
        shuffle=False
    )
    datasets.setup_univariate_batch_gen(batch_gen_valid, 
        dataset, variable, batch_size=batch_size)
    model_func = models.rnn2_model if model_type == "convgru" \
        else models.rnn3_model
    model = models.init_model(batch_gen_valid, model_func=model_func)
    if weight_fn is not None:
        model.load_weights(weight_fn)    

    eval_results = model.evaluate(batch_gen_valid)
    print(eval_results)


def train(
    comp_dir="w4c-core-stage-1",
    regions=None,
    dataset="CTTH",
    variable="temperature",
    batch_size=32,
    model_type="resgru",
    weight_fn=None
):
    if regions is None:
        if "core" in comp_dir:
            regions = ["R1", "R2", "R3"]
        else:
            regions = ["R4", "R5", "R6"]

    batch_gen_train = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="training"
    )
    batch_gen_valid = datasets.BatchGenerator(
        comp_dir=comp_dir,
        regions=regions,
        data_subset="validation",
        augment=False,
        shuffle=False
    )
    datasets.setup_univariate_batch_gen(batch_gen_train, 
        dataset, variable, batch_size=batch_size)
    datasets.setup_univariate_batch_gen(batch_gen_valid, 
        dataset, variable, batch_size=batch_size)
    model_func = models.rnn2_model if model_type == "convgru" \
        else models.rnn3_model
    model = models.init_model(batch_gen_valid, model_func=model_func)
    print(batch_gen_train[0][0].shape, model.input_shape)
    print(batch_gen_train[0][1][0].shape, model.output_shape)
    models.train_model(model, batch_gen_train, batch_gen_valid,
        weight_fn=weight_fn)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str,
        help="submit / evaluate / train")
    parser.add_argument('--comp_dir', type=str,
        help="Directory where the data are located")
    parser.add_argument('--regions', type=str, 
        help="Comma-separated list or regions, default all regions for comp_dir")
    parser.add_argument('--submission_dir', type=str, default="",
        help="Directory to save the results in, will be created if needed")
    parser.add_argument('--batch_size', type=int, default=32,
        help="Batch size for training / evaluation")
    parser.add_argument('--dataset', type=str, default="",
        help="Dataset for training / evaluation")
    parser.add_argument('--variable', type=str, default="",
        help="Variable for training / evaluation")
    parser.add_argument('--weights', type=str, default="",
        help="Model weight file for training / evaluation")
    parser.add_argument('--model', type=str, default="resgru",
        help="Model type for training / evaluation, either 'convgru' or 'resgru'")

    args = parser.parse_args()
    mode = args.mode
    regions = args.regions
    if not regions:
        regions = None
    else:
        regions = regions.split(",")
    comp_dir = args.comp_dir

    if mode == "submit":        
        submission_dir = args.submission_dir
        assert(submission_dir != "")
        generate_predictions(submission_dir,
            comp_dir=comp_dir, regions=regions)
    elif mode in ["evaluate", "train"]:
        batch_size = args.batch_size
        dataset = args.dataset
        variable = args.variable
        weight_fn = args.weights
        model_type = args.model
        assert(dataset in ["CTTH", "CRR", "ASII", "CMA"])
        assert(variable in ["temperature", "crr_intensity",
            "asii_turb_trop_prob", "cma"])
        if mode == "evaluate":
            evaluate(comp_dir=comp_dir, regions=regions, dataset=dataset,
                variable=variable, batch_size=batch_size, weight_fn=weight_fn,
                model_type=model_type)
        else:
            train(comp_dir=comp_dir, regions=regions, dataset=dataset,
                variable=variable, batch_size=batch_size, weight_fn=weight_fn,
                model_type=model_type)

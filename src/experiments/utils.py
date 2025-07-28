import argparse

def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse experiment settings.")

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input json.",
    )

    parser.add_argument(
        "--environment",
        type=str,
        choices=["local", "hpc"],
        required=False,
        default="hpc",
        help="Choose the environment: 'local' or 'hpc'.",
    )

    parser.add_argument(
            "--model",
            type=str,
            choices=["qwenaudio", "mullama"],
            required=True
            )

    parser.add_argument("--range", type=int, required=False, default=None,
            help="Parallelization parameter. Specifies how many questions we cover per run. If not provided, we process the full json file.")

    parser.add_argument("--index", type=int, required=False, default=0,
            help="Parallelization parameter. Specifies where we start in this run. If not provided, we start by index 0.")


    args = parser.parse_args()

    return args
